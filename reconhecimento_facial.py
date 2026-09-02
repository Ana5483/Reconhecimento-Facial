from contextlib import contextmanager
from datetime import datetime
import argparse
import csv
import logging
import os
import random
import shutil
import sqlite3
import sys
import threading
import time
import unicodedata

import numpy as np

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp|fflags;nobuffer|max_delay;500000"
)

try:
    import cv2
except ImportError:
    print("[ERRO CRITICO] OpenCV nao encontrado.")
    print("Instale com: pip install opencv-python opencv-contrib-python")
    sys.exit(1)

try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_DISPONIVEL = True
except ImportError:
    INSIGHTFACE_DISPONIVEL = False

CONFIG = {
    "BASE_DIR": "database",
    "RTSP_URL": "rtsp://admin:%40nvd%401234%3F@192.168.3.25:554/cam/realmonitor?channel=1&subtype=0",

    "INSIGHTFACE_PACK": "buffalo_l",
    "USE_GPU": False,
    "DET_SIZE": 640,
    "DET_SCORE_MIN": 0.62,
    "MIN_FACE_SIZE": 20,
    "PROCESS_EVERY_N_FRAMES": 2,
    "QUALIDADE_MIN_CADASTRO": 45.0,
    "NITIDEZ_MIN": 25.0,
    "FRONTALIDADE_MAX": 0.80,
    "FRONTALIDADE_CADASTRO": 0.45,
    "FRAMES_MIN_CADASTRO": 6,
    "EMBEDDINGS_POR_PESSOA": 5,
    "CADASTRO_FALLBACK_SEGUNDOS": 2.5,
    "CADASTRO_FALLBACK_FRAMES": 10,
    "SALVAR_LADO_ALVO": 340,
    "SALVAR_JPEG_QUALIDADE": 97,
    "SALVAR_NITIDEZ_FORCA": 0.45,
    "SALVAR_CLAHE_CLIP": 1.6,

    # Reconhecimento: o match so trava com limiar + folga para a 2a pessoa
    # mais parecida + a mesma identidade repetida em varios frames bons.
    # Na duvida, cria ID novo em vez de fundir duas pessoas.
    "SIM_RECONHECER": 0.46,
    "SIM_ZONA_CINZA": 0.38,
    "SIM_APRENDER": 0.60,
    "SIM_MARGEM_2A": 0.08,
    "RECOG_FRAMES_MIN": 4,
    "RECOG_QUALIDADE_MIN": 32.0,
    "RECOG_JANELA": 6,
    "RECOG_CONSISTENCIA": 3,

    # Genero: so vota o frame de alta qualidade (o modelo erra em rosto
    # pequeno ou torto). Sem quorum de votos -> "nao_reconhecido".
    "GENERO_VOTOS_MIN": 5,
    "GENERO_MAIORIA_MIN": 0.85,
    "GENERO_FRAME_LADO_MIN": 46,
    "GENERO_FRAME_FRONTAL_MAX": 0.26,
    "GENERO_FRAME_NITIDEZ_MIN": 90,
    "GENERO_FRAME_DET_MIN": 0.72,

    # Anti-spam 
    "COOLDOWN_RECONHECIMENTO": 300,  # segundos entre registros da mesma pessoa
    "TRACK_IOU_MIN": 0.28,
    "TRACK_MAX_MISSING": 4.0,

    # Interface 
    "DISPLAY_WIDTH": 960,
    "DISPLAY_HEIGHT": 640,
    "PANEL_WIDTH": 380,
    "RECONNECT_INTERVAL": 3.0,
    "HISTORICO_PAINEL": 10,

    "NOME_INDEFINIDO": "nao_reconhecido",

    # Cores (BGR) 
    "C_BG":        (22, 20, 28),
    "C_CARD":      (36, 32, 46),
    "C_BORDA":     (62, 56, 76),
    "C_PRIMARY":   (255, 220, 90),
    "C_TEXTO":     (238, 238, 242),
    "C_MUTED":     (150, 146, 160),
    "C_VERDE":     (90, 222, 120),
    "C_VERMELHO":  (72, 72, 235),
    "C_AMARELO":   (60, 190, 250),
    "C_CIANO":     (220, 200, 70),
}

def _validar_config():
    erros = []
    if CONFIG["GENERO_VOTOS_MIN"] > CONFIG["FRAMES_MIN_CADASTRO"]:
        erros.append(
            f"GENERO_VOTOS_MIN ({CONFIG['GENERO_VOTOS_MIN']}) nao pode ser maior que "
            f"FRAMES_MIN_CADASTRO ({CONFIG['FRAMES_MIN_CADASTRO']}): o sexo nunca "
            f"atingiria quorum e toda pessoa nasceria como 'nao_reconhecido'.")
    if CONFIG["FRONTALIDADE_CADASTRO"] > CONFIG["FRONTALIDADE_MAX"]:
        erros.append("FRONTALIDADE_CADASTRO tem de ser <= FRONTALIDADE_MAX.")
    if CONFIG["SIM_ZONA_CINZA"] > CONFIG["SIM_RECONHECER"]:
        erros.append("SIM_ZONA_CINZA tem de ser <= SIM_RECONHECER.")
    if CONFIG["SIM_APRENDER"] < CONFIG["SIM_RECONHECER"]:
        erros.append("SIM_APRENDER tem de ser >= SIM_RECONHECER.")
    if CONFIG["MIN_FACE_SIZE"] < 18:
        erros.append("MIN_FACE_SIZE abaixo de 18px: o modelo deixa de distinguir "
                     "pessoas e passa a fundir identidades.")
    if erros:
        raise ValueError("CONFIG inconsistente:\n  - " + "\n  - ".join(erros))


_validar_config()

BASE_DIR = CONFIG["BASE_DIR"]
REGISTRO_DIR = os.path.join(BASE_DIR, "registro")        
COMPARACOES_DIR = os.path.join(BASE_DIR, "comparacoes") 
LOGS_DIR = os.path.join(BASE_DIR, "logs")
DB_PATH = os.path.join(BASE_DIR, "system.db")
PASTAS_ATIVAS = [BASE_DIR, REGISTRO_DIR, COMPARACOES_DIR, LOGS_DIR]
PASTAS_OBSOLETAS = ["faces", "detections", "capturas", "temp", "cache"]

for _p in PASTAS_ATIVAS:
    os.makedirs(_p, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOGS_DIR, "system.log"),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    encoding="utf-8",
)

NOMES_FEMININOS = [
    "Maria Clara", "Ana Beatriz", "Julia Fernandes", "Camila Rocha",
    "Fernanda Oliveira", "Beatriz Santos", "Patricia Alves", "Isabela Pereira",
    "Carolina Dias", "Mariana Souza", "Gabriela Gomes", "Aline Cardoso",
    "Leticia Barbosa", "Vanessa Teixeira", "Sandra Cavalcante", "Rita Mendes",
    "Larissa Correia", "Gisele Soares", "Adriana Monteiro", "Alessandra Pinto",
    "Amanda Lopes", "Angelica Medeiros", "Bruna Macedo", "Cecilia Ribeiro",
    "Daniela Fonseca", "Elena Campos", "Fabiana Barreto", "Helena Guedes",
    "Luiza Martins", "Natalia Freitas", "Renata Vieira", "Sofia Andrade",
]

NOMES_MASCULINOS = [
    "Joao Pedro", "Carlos Eduardo", "Paulo Henrique", "Marcos Vinicius",
    "Andre Ferreira", "Daniel Martins", "Rafael Pereira", "Roberto Dias",
    "Ricardo Rocha", "Fernando Gomes", "Felipe Souza", "Gustavo Cardoso",
    "Lucas Barbosa", "Bruno Teixeira", "Thiago Cavalcante", "Diego Mendes",
    "Alan Correia", "Eduardo Soares", "Sergio Monteiro", "Marcelo Pinto",
    "Jorge Lopes", "Jose Medeiros", "Antonio Macedo", "Cesar Ribeiro",
    "Evandro Campos", "Francisco Barreto", "Gilson Neves", "Henrique Guedes",
    "Leonardo Freitas", "Murilo Andrade", "Rodrigo Vieira", "Vitor Nogueira",
]

_ACENTOS = str.maketrans(
    "áàãâäéèêëíìîïóòõôöúùûüçñÁÀÃÂÄÉÈÊËÍÌÎÏÓÒÕÔÖÚÙÛÜÇÑ",
    "aaaaaeeeeiiiiooooouuuucnAAAAAEEEEIIIIOOOOOUUUUCN",
)

def sem_acento(texto):
    """OpenCV nao renderiza acentos: converte para ASCII antes de desenhar."""
    if not texto:
        return ""
    return unicodedata.normalize("NFC", str(texto)).translate(_ACENTOS)

def letterbox(img, largura, altura, cor_fundo=(30, 27, 36)):
    fundo = np.full((altura, largura, 3), cor_fundo, dtype=np.uint8)
    if img is None or img.size == 0:
        return fundo
    ih, iw = img.shape[:2]
    if iw <= 0 or ih <= 0:
        return fundo
    escala = min(largura / iw, altura / ih)
    nw, nh = max(1, int(iw * escala)), max(1, int(ih * escala))
    interp = cv2.INTER_AREA if escala < 1 else cv2.INTER_LINEAR
    redim = cv2.resize(img, (nw, nh), interpolation=interp)
    ox, oy = (largura - nw) // 2, (altura - nh) // 2
    fundo[oy:oy + nh, ox:ox + nw] = redim
    return fundo

_CLAHE_SALVAR = cv2.createCLAHE(
    clipLimit=float(CONFIG["SALVAR_CLAHE_CLIP"]), tileGridSize=(8, 8))

def melhorar_para_salvar(crop):
    if crop is None or crop.size == 0:
        return crop
    img = np.ascontiguousarray(crop)
    h, w = img.shape[:2]
    if min(h, w) < 6:
        return img
    try:
        img = cv2.bilateralFilter(img, d=5, sigmaColor=24, sigmaSpace=24)

        alvo = int(CONFIG["SALVAR_LADO_ALVO"])
        escala = alvo / float(min(h, w))
        if escala > 1.02:
            novo = (int(round(w * escala)), int(round(h * escala)))
            img = cv2.resize(img, novo, interpolation=cv2.INTER_LANCZOS4)
        elif escala < 0.98:
            novo = (max(1, int(round(w * escala))), max(1, int(round(h * escala))))
            img = cv2.resize(img, novo, interpolation=cv2.INTER_AREA)

        forca = float(CONFIG["SALVAR_NITIDEZ_FORCA"])
        if forca > 0:
            blur = cv2.GaussianBlur(img, (0, 0), sigmaX=1.6)
            img = cv2.addWeighted(img, 1.0 + forca, blur, -forca, 0)

        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = _CLAHE_SALVAR.apply(l)
        img = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    except cv2.error as e:
        logging.warning("melhorar_para_salvar falhou, salvando recorte cru: %s", e)
        return np.ascontiguousarray(crop)

    return img

def salvar_imagem_rosto(caminho, crop):
    """Aplica o tratamento e grava em disco. Retorna o caminho ou '' em erro."""
    img = melhorar_para_salvar(crop)
    if img is None or img.size == 0:
        return ""
    ok = cv2.imwrite(caminho, img,
                     [cv2.IMWRITE_JPEG_QUALITY, int(CONFIG["SALVAR_JPEG_QUALIDADE"])])
    return caminho if ok else ""

def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    return inter / float(area_a + area_b - inter + 1e-6)

def normalizar_vetor(v):
    v = np.asarray(v, dtype=np.float32).flatten()
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v

def frontalidade(kps):
    if kps is None or len(kps) < 3:
        return 1.0
    olho_e, olho_d, nariz = kps[0], kps[1], kps[2]
    dist_olhos = float(np.linalg.norm(olho_d - olho_e))
    if dist_olhos < 1e-3:
        return 1.0
    centro_x = (olho_e[0] + olho_d[0]) / 2.0
    desvio = abs(float(nariz[0]) - centro_x) / dist_olhos
    return float(desvio)

def qualidade_rosto(crop, lado_rosto, det_score, desvio_frontal):
    if crop is None or crop.size == 0:
        return 0.0, 0.0
    try:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        nitidez = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brilho = float(np.mean(gray))

        s_nitidez = min(1.0, nitidez / 700.0)
        s_tamanho = max(0.0, min(1.0, (lado_rosto - 18.0) / 72.0))
        s_brilho = max(0.0, 1.0 - abs(brilho - 128.0) / 128.0)
        s_frontal = max(0.0, 1.0 - desvio_frontal / 0.5)
        s_det = max(0.0, min(1.0, (det_score - 0.4) / 0.6))

        score = 100.0 * (
            0.34 * s_nitidez
            + 0.27 * s_tamanho
            + 0.08 * s_brilho
            + 0.25 * s_frontal
            + 0.06 * s_det
        )
        return float(score), nitidez
    except Exception:
        return 0.0, 0.0

def retangulo_arredondado(img, pt1, pt2, cor, espessura=1, raio=8):
    x1, y1 = pt1
    x2, y2 = pt2
    raio = max(1, min(raio, abs(x2 - x1) // 2, abs(y2 - y1) // 2))
    if espessura < 0:
        cv2.rectangle(img, (x1 + raio, y1), (x2 - raio, y2), cor, -1)
        cv2.rectangle(img, (x1, y1 + raio), (x2, y2 - raio), cor, -1)
        for cx, cy in ((x1 + raio, y1 + raio), (x2 - raio, y1 + raio),
                       (x1 + raio, y2 - raio), (x2 - raio, y2 - raio)):
            cv2.circle(img, (cx, cy), raio, cor, -1, cv2.LINE_AA)
    else:
        cv2.line(img, (x1 + raio, y1), (x2 - raio, y1), cor, espessura, cv2.LINE_AA)
        cv2.line(img, (x1 + raio, y2), (x2 - raio, y2), cor, espessura, cv2.LINE_AA)
        cv2.line(img, (x1, y1 + raio), (x1, y2 - raio), cor, espessura, cv2.LINE_AA)
        cv2.line(img, (x2, y1 + raio), (x2, y2 - raio), cor, espessura, cv2.LINE_AA)
        cv2.ellipse(img, (x1 + raio, y1 + raio), (raio, raio), 180, 0, 90, cor, espessura, cv2.LINE_AA)
        cv2.ellipse(img, (x2 - raio, y1 + raio), (raio, raio), 270, 0, 90, cor, espessura, cv2.LINE_AA)
        cv2.ellipse(img, (x1 + raio, y2 - raio), (raio, raio), 90, 0, 90, cor, espessura, cv2.LINE_AA)
        cv2.ellipse(img, (x2 - raio, y2 - raio), (raio, raio), 0, 0, 90, cor, espessura, cv2.LINE_AA)

def texto(img, msg, org, escala=0.4, cor=(240, 240, 240), espessura=1):
    cv2.putText(img, sem_acento(msg), org, cv2.FONT_HERSHEY_SIMPLEX,
                escala, cor, espessura, cv2.LINE_AA)

def texto_limitado(msg, largura_px, escala=0.4, espessura=1):
    """Trunca o texto com reticencias para caber em largura_px."""
    msg = sem_acento(msg)
    (tw, _), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, escala, espessura)
    if tw <= largura_px:
        return msg
    while len(msg) > 1:
        msg = msg[:-1]
        (tw, _), _ = cv2.getTextSize(msg + "...", cv2.FONT_HERSHEY_SIMPLEX, escala, espessura)
        if tw <= largura_px:
            return msg + "..."
    return msg

def _pausar(mensagem="\nPressione ENTER para voltar..."):
    try:
        if sys.stdin is not None and sys.stdin.isatty():
            input(mensagem)
    except (EOFError, OSError):
        pass

class ImageCache:
    def __init__(self, max_entries=120):
        self.cache = {}
        self.max_entries = max_entries

    def get(self, filepath):
        if not filepath or not os.path.exists(filepath):
            return None
        try:
            chave = (filepath, os.path.getmtime(filepath))
            if chave in self.cache:
                return self.cache[chave]
            img = cv2.imread(filepath)
            if img is None:
                return None
            if len(self.cache) >= self.max_entries:
                self.cache.pop(next(iter(self.cache)))
            self.cache[chave] = img
            return img
        except Exception as e:
            logging.error("Falha ao carregar imagem %s: %s", filepath, e)
            return None

class DatabaseManager:
    def __init__(self, db_path):
        self.db_path = db_path
        self._lock = threading.RLock()
        self._init_db()

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path, timeout=15)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self):
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        self._converter_esquema_v4()
        self._criar_tabelas()
        self._importar_dados_v4()
        logging.info("Banco de dados pronto: %s", self.db_path)

    @staticmethod
    def _tabelas(conn):
        return {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}

    @staticmethod
    def _colunas(conn, tabela):
        return {r[1] for r in conn.execute(f"PRAGMA table_info({tabela})").fetchall()}

    def _converter_esquema_v4(self):
        with self._lock, self._conn() as conn:
            tabelas = self._tabelas(conn)
            if "pessoas" not in tabelas:
                return
            if "foto_referencia" in self._colunas(conn, "pessoas"):
                return  

            # legacy_alter_table=ON impede que o RENAME reescreva as chaves
            # estrangeiras das outras tabelas para apontar para 'pessoas_v4'.
            conn.execute("PRAGMA foreign_keys=OFF")
            conn.execute("PRAGMA legacy_alter_table=ON")
            if "pessoas_v4" in tabelas:
                conn.execute("DROP TABLE pessoas_v4")
            conn.execute("ALTER TABLE pessoas RENAME TO pessoas_v4")
            for antiga, nova in (("registros", "registros_v4"),
                                 ("reconhecimentos", "reconhecimentos_v4")):
                if antiga in tabelas:
                    if nova in tabelas:
                        conn.execute(f"DROP TABLE {nova}")
                    conn.execute(f"ALTER TABLE {antiga} RENAME TO {nova}")
            conn.execute("PRAGMA legacy_alter_table=OFF")

        destino = os.path.join(
            os.path.dirname(self.db_path) or ".",
            f"system_v4_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db")
        try:
            shutil.copy2(self.db_path, destino)
            print(f"[MIGRACAO] Backup do banco v4 salvo em: {destino}")
        except Exception as e:
            logging.warning("Nao foi possivel salvar backup do banco v4: %s", e)
        logging.info("Esquema v4 renomeado; iniciando migracao para v5.")

    def _criar_tabelas(self):
        with self._lock, self._conn() as conn:
            c = conn.cursor()
            c.execute("""
                CREATE TABLE IF NOT EXISTS pessoas (
                    id                TEXT PRIMARY KEY,
                    nome              TEXT NOT NULL,
                    genero            TEXT NOT NULL DEFAULT 'Indeterminado',
                    embedding         BLOB NOT NULL,
                    data_cadastro     TEXT NOT NULL,
                    ultima_deteccao   TEXT,
                    foto_referencia   TEXT DEFAULT '',
                    total_aparicoes   INTEGER DEFAULT 1,
                    status            TEXT DEFAULT 'Ativo'
                )
            """)
            c.execute("""
                CREATE TABLE IF NOT EXISTS templates (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    pessoa_id  TEXT NOT NULL,
                    vetor      BLOB NOT NULL,
                    qualidade  REAL DEFAULT 0,
                    criado_em  TEXT NOT NULL,
                    FOREIGN KEY (pessoa_id) REFERENCES pessoas(id) ON DELETE CASCADE
                )
            """)
            c.execute("""
                CREATE TABLE IF NOT EXISTS historico (
                    id             INTEGER PRIMARY KEY AUTOINCREMENT,
                    pessoa_id      TEXT NOT NULL,
                    tipo           TEXT NOT NULL,
                    data_hora      TEXT NOT NULL,
                    caminho_imagem TEXT,
                    similaridade   REAL DEFAULT 0,
                    FOREIGN KEY (pessoa_id) REFERENCES pessoas(id) ON DELETE CASCADE
                )
            """)
            c.execute("CREATE INDEX IF NOT EXISTS idx_hist_pessoa ON historico(pessoa_id)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_hist_data ON historico(data_hora)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_tpl_pessoa ON templates(pessoa_id)")

    def _importar_dados_v4(self):
        """
        Copia pessoas e historico da v4. Os embeddings antigos sao descartados
        (nao sao comparaveis com os da v5): essas pessoas entram como 'Legado'
        e sao recadastradas na proxima passagem.
        """
        with self._lock, self._conn() as conn:
            tabelas = self._tabelas(conn)
            if "pessoas_v4" not in tabelas:
                return
            c = conn.cursor()
            if c.execute("SELECT COUNT(*) FROM pessoas").fetchone()[0] > 0:
                return

            colunas = self._colunas(conn, "pessoas_v4")
            col_nome = "nome_completo" if "nome_completo" in colunas else "nome"
            col_foto = ("caminho_foto_referencia"
                        if "caminho_foto_referencia" in colunas else "''")

            pessoas = c.execute(
                f"SELECT id, {col_nome}, genero, data_cadastro, ultima_deteccao,"
                f" {col_foto} FROM pessoas_v4"
            ).fetchall()

            vazio = np.zeros(1, dtype=np.float32).tobytes()
            for pid, nome, genero, cadastro, ultima, foto in pessoas:
                if not foto or not os.path.exists(str(foto)):
                    candidato = os.path.join(REGISTRO_DIR, str(pid), "referencia.jpg")
                    foto = candidato if os.path.exists(candidato) else ""
                c.execute(
                    "INSERT OR IGNORE INTO pessoas (id, nome, genero, embedding,"
                    " data_cadastro, ultima_deteccao, foto_referencia, total_aparicoes, status)"
                    " VALUES (?,?,?,?,?,?,?,?,?)",
                    (pid, nome or CONFIG["NOME_INDEFINIDO"], genero or "Indeterminado",
                     vazio, cadastro or "", ultima, foto, 1, "Legado"),
                )

            ids_validos = {r[0] for r in c.execute("SELECT id FROM pessoas").fetchall()}
            eventos = 0
            for tabela, tipo in (("registros_v4", "REGISTRO"),
                                 ("reconhecimentos_v4", "RECONHECIMENTO")):
                if tabela not in tabelas:
                    continue
                try:
                    linhas = c.execute(
                        f"SELECT pessoa_id, data_hora, caminho_imagem, confianca"
                        f" FROM {tabela}").fetchall()
                except sqlite3.Error:
                    continue
                for pid, data_hora, img, conf in linhas:
                    if pid not in ids_validos:
                        continue
                    c.execute(
                        "INSERT INTO historico (pessoa_id, tipo, data_hora,"
                        " caminho_imagem, similaridade) VALUES (?,?,?,?,?)",
                        (pid, tipo, data_hora, img, (conf or 0) / 100.0),
                    )
                    eventos += 1

        if pessoas:
            print(f"[MIGRACAO] {len(pessoas)} pessoa(s) e {eventos} evento(s) da v4 "
                  f"importados. As pessoas antigas entraram como 'Legado' porque os "
                  f"vetores da v4 nao sao confiaveis; elas serao recadastradas "
                  f"automaticamente na proxima passagem.")
            logging.info("Migracao v4->v5: %d pessoas (Legado), %d eventos.",
                         len(pessoas), eventos)

    def proximo_id(self):
        """ID sequencial baseado no MAIOR ja emitido (nunca reutiliza numero)."""
        with self._lock, self._conn() as conn:
            maior = 0
            for (pid,) in conn.execute("SELECT id FROM pessoas").fetchall():
                if isinstance(pid, str) and pid.startswith("ID_"):
                    sufixo = pid[3:]
                    if sufixo.isdigit():
                        maior = max(maior, int(sufixo))
            return f"ID_{maior + 1:03d}"

    def nomes_em_uso(self):
        with self._lock, self._conn() as conn:
            return {r[0] for r in conn.execute("SELECT nome FROM pessoas").fetchall()}


    def cadastrar_pessoa(self, pessoa_id, nome, genero, embedding, foto_ref, qualidade):
        agora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        blob = normalizar_vetor(embedding).tobytes()
        with self._lock, self._conn() as conn:
            try:
                conn.execute(
                    "INSERT INTO pessoas (id, nome, genero, embedding, data_cadastro,"
                    " ultima_deteccao, foto_referencia, total_aparicoes, status)"
                    " VALUES (?,?,?,?,?,?,?,?,?)",
                    (pessoa_id, nome, genero, blob, agora, agora, foto_ref, 1, "Ativo"),
                )
                conn.execute(
                    "INSERT INTO templates (pessoa_id, vetor, qualidade, criado_em)"
                    " VALUES (?,?,?,?)", (pessoa_id, blob, qualidade, agora)
                )
                conn.commit()
            except sqlite3.IntegrityError:
                logging.warning("Duplicacao bloqueada no cadastro de %s", pessoa_id)
                return False
        logging.info("CADASTRO: %s - %s (%s) q=%.1f", pessoa_id, nome, genero, qualidade)
        return True

    def definir_foto_referencia(self, pessoa_id, caminho):
        with self._lock, self._conn() as conn:
            conn.execute(
                "UPDATE pessoas SET foto_referencia=? WHERE id=? AND"
                " (foto_referencia IS NULL OR foto_referencia='')",
                (caminho, pessoa_id),
            )
            conn.commit()

    def resolver_genero_indefinido(self, pessoa_id, genero, nome):
        """
        So altera o sexo se ele AINDA estiver indeterminado.
        Um sexo ja confirmado nunca e sobrescrito -- evita trocar o sexo da pessoa.
        """
        if genero not in ("Masculino", "Feminino"):
            return False
        with self._lock, self._conn() as conn:
            cur = conn.execute(
                "UPDATE pessoas SET genero=?, nome=? WHERE id=? AND genero='Indeterminado'",
                (genero, nome, pessoa_id),
            )
            conn.commit()
            alterou = cur.rowcount > 0
        if alterou:
            logging.info("GENERO RESOLVIDO: %s -> %s (%s)", pessoa_id, genero, nome)
        return alterou

    def forcar_identidade(self, pessoa_id, genero, nome):
        """Correcao MANUAL: sobrescreve sexo e nome de uma pessoa ja cadastrada."""
        with self._lock, self._conn() as conn:
            cur = conn.execute(
                "UPDATE pessoas SET genero=?, nome=? WHERE id=?",
                (genero, nome, pessoa_id),
            )
            conn.commit()
        if cur.rowcount > 0:
            logging.info("CORRECAO MANUAL: %s -> %s (%s)", pessoa_id, nome, genero)
        return cur.rowcount > 0

    def adicionar_template(self, pessoa_id, embedding, qualidade):
        """Guarda um template extra, mantendo apenas os melhores por pessoa."""
        agora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        blob = normalizar_vetor(embedding).tobytes()
        limite = CONFIG["EMBEDDINGS_POR_PESSOA"]
        with self._lock, self._conn() as conn:
            conn.execute(
                "INSERT INTO templates (pessoa_id, vetor, qualidade, criado_em) VALUES (?,?,?,?)",
                (pessoa_id, blob, qualidade, agora),
            )
            conn.execute(
                "DELETE FROM templates WHERE id IN ("
                "  SELECT id FROM templates WHERE pessoa_id=?"
                "  ORDER BY qualidade DESC, id ASC LIMIT -1 OFFSET ?)",
                (pessoa_id, limite),
            )
            conn.commit()

    def carregar_galeria(self):
        """Carrega pessoas + matriz de templates para busca vetorial rapida."""
        galeria = []
        with self._lock, self._conn() as conn:
            pessoas = conn.execute(
                "SELECT id, nome, genero, foto_referencia, data_cadastro, ultima_deteccao,"
                " total_aparicoes FROM pessoas WHERE status='Ativo' ORDER BY id"
            ).fetchall()
            for pid, nome, genero, foto, cadastro, ultima, aparicoes in pessoas:
                vetores = []
                for (blob,) in conn.execute(
                    "SELECT vetor FROM templates WHERE pessoa_id=?", (pid,)
                ).fetchall():
                    v = np.frombuffer(blob, dtype=np.float32)
                    if v.size and not np.all(v == 0):
                        vetores.append(normalizar_vetor(v))
                if not vetores:
                    continue
                galeria.append({
                    "id": pid,
                    "nome": nome,
                    "genero": genero or "Indeterminado",
                    "foto_referencia": foto or "",
                    "data_cadastro": cadastro,
                    "ultima_deteccao": ultima,
                    "total_aparicoes": aparicoes or 1,
                    "templates": np.vstack(vetores),
                })
        return galeria

    def obter_pessoa(self, pessoa_id):
        with self._lock, self._conn() as conn:
            row = conn.execute(
                "SELECT id, nome, genero, foto_referencia, data_cadastro, ultima_deteccao,"
                " total_aparicoes FROM pessoas WHERE id=?", (pessoa_id,)
            ).fetchone()
        if not row:
            return None
        return {
            "id": row[0], "nome": row[1], "genero": row[2] or "Indeterminado",
            "foto_referencia": row[3] or "", "data_cadastro": row[4],
            "ultima_deteccao": row[5], "total_aparicoes": row[6] or 1,
        }

    def registrar_evento(self, pessoa_id, tipo, caminho_imagem, similaridade):
        agora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._lock, self._conn() as conn:
            conn.execute(
                "INSERT INTO historico (pessoa_id, tipo, data_hora, caminho_imagem, similaridade)"
                " VALUES (?,?,?,?,?)",
                (pessoa_id, tipo, agora, caminho_imagem, float(similaridade)),
            )
            conn.execute(
                "UPDATE pessoas SET ultima_deteccao=?,"
                " total_aparicoes=COALESCE(total_aparicoes,0)+1 WHERE id=?",
                (agora, pessoa_id),
            )
            conn.commit()
        logging.info("%s: %s sim=%.3f", tipo, pessoa_id, similaridade)
        return agora

    def tem_registro(self, pessoa_id):
        with self._lock, self._conn() as conn:
            row = conn.execute(
                "SELECT 1 FROM historico WHERE pessoa_id=? AND tipo='REGISTRO' LIMIT 1",
                (pessoa_id,),
            ).fetchone()
        return row is not None

    def ultimo_reconhecimento(self, pessoa_id):
        with self._lock, self._conn() as conn:
            return conn.execute(
                "SELECT data_hora, caminho_imagem, similaridade FROM historico"
                " WHERE pessoa_id=? AND tipo='RECONHECIMENTO' ORDER BY id DESC LIMIT 1",
                (pessoa_id,),
            ).fetchone()

    def historico_recente(self, limite=10):
        with self._lock, self._conn() as conn:
            return conn.execute(
                "SELECT h.data_hora, h.tipo, h.pessoa_id, p.nome, p.genero, h.similaridade"
                " FROM historico h JOIN pessoas p ON p.id=h.pessoa_id"
                " ORDER BY h.id DESC LIMIT ?", (limite,)
            ).fetchall()

    def historico_completo(self):
        with self._lock, self._conn() as conn:
            return conn.execute(
                "SELECT h.data_hora, h.tipo, h.pessoa_id, p.nome, p.genero,"
                " h.similaridade, h.caminho_imagem"
                " FROM historico h JOIN pessoas p ON p.id=h.pessoa_id"
                " ORDER BY h.id DESC"
            ).fetchall()

    def estatisticas(self):
        with self._lock, self._conn() as conn:
            c = conn.cursor()
            cadastrados = c.execute(
                "SELECT COUNT(*) FROM pessoas WHERE status='Ativo'").fetchone()[0]
            registros = c.execute(
                "SELECT COUNT(*) FROM historico WHERE tipo='REGISTRO'").fetchone()[0]
            reconhecimentos = c.execute(
                "SELECT COUNT(*) FROM historico WHERE tipo='RECONHECIMENTO'").fetchone()[0]
            hoje = c.execute(
                "SELECT COUNT(DISTINCT pessoa_id) FROM historico"
                " WHERE DATE(data_hora)=DATE('now','localtime')").fetchone()[0]
        return cadastrados, registros, reconhecimentos, hoje

    def caminhos_de_imagem(self):
        with self._lock, self._conn() as conn:
            caminhos = {r[0] for r in conn.execute(
                "SELECT foto_referencia FROM pessoas WHERE foto_referencia<>''").fetchall()}
            caminhos |= {r[0] for r in conn.execute(
                "SELECT caminho_imagem FROM historico WHERE caminho_imagem IS NOT NULL"
                " AND caminho_imagem<>''").fetchall()}
        return caminhos

class RTSPCameraStream:
    def __init__(self, rtsp_url):
        self.rtsp_url = rtsp_url
        self.cap = None
        self.frame = None
        self.is_connected = False
        self.stopped = False
        self.lock = threading.Lock()
        self.thread = None

    def start(self):
        self.stopped = False
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        return self

    def _connect(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        try:
            self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.is_connected = self.cap.isOpened()
        except Exception as e:
            logging.error("Erro ao conectar na camera: %s", e)
            self.is_connected = False

    def _update(self):
        while not self.stopped:
            if not self.is_connected or self.cap is None or not self.cap.isOpened():
                self._connect()
                if not self.is_connected:
                    time.sleep(CONFIG["RECONNECT_INTERVAL"])
                    continue
            try:
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    self.is_connected = False
                    continue
                with self.lock:
                    self.frame = frame
                    self.is_connected = True
            except Exception as e:
                logging.error("Erro de leitura da camera: %s", e)
                self.is_connected = False

    def read(self):
        with self.lock:
            if self.frame is not None:
                return self.is_connected, self.frame.copy()
            return self.is_connected, None

    def stop(self):
        self.stopped = True
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.5)
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass

class FaceEngine:
    def __init__(self):
        self.app = None
        self.disponivel = False
        self.nome_modelo = "INDISPONIVEL"

        if not INSIGHTFACE_DISPONIVEL:
            print("[ERRO CRITICO] InsightFace nao encontrado.")
            print("Instale com: pip install insightface onnxruntime")
            logging.error("InsightFace ausente - reconhecimento desabilitado.")
            return

        try:
            providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                         if CONFIG["USE_GPU"] else ["CPUExecutionProvider"])
            self.app = FaceAnalysis(
                name=CONFIG["INSIGHTFACE_PACK"],
                providers=providers,
                allowed_modules=["detection", "recognition", "genderage"],
            )
            self.app.prepare(
                ctx_id=(0 if CONFIG["USE_GPU"] else -1),
                det_size=(CONFIG["DET_SIZE"], CONFIG["DET_SIZE"]),
                det_thresh=CONFIG["DET_SCORE_MIN"],
            )
            self.disponivel = True
            self.nome_modelo = CONFIG["INSIGHTFACE_PACK"].upper()
            logging.info("InsightFace %s carregado (GPU=%s).",
                         self.nome_modelo, CONFIG["USE_GPU"])
        except Exception as e:
            logging.error("Falha ao carregar InsightFace: %s", e)
            print(f"[ERRO CRITICO] Nao foi possivel carregar o InsightFace: {e}")

    def analisar(self, frame):
        """Retorna a lista de rostos validos com bbox, embedding, genero e qualidade."""
        if not self.disponivel or frame is None:
            return []

        try:
            faces = self.app.get(frame)
        except Exception as e:
            logging.error("Erro na analise facial: %s", e)
            return []

        h_frame, w_frame = frame.shape[:2]
        resultado = []

        for f in faces:
            det_score = float(getattr(f, "det_score", 0.0))
            if det_score < CONFIG["DET_SCORE_MIN"]:
                continue

            x1, y1, x2, y2 = [int(v) for v in f.bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_frame, x2), min(h_frame, y2)
            lw, lh = x2 - x1, y2 - y1
            lado = min(lw, lh)
            if lado < CONFIG["MIN_FACE_SIZE"]:
                continue

            kps = getattr(f, "kps", None)
            desvio = frontalidade(np.asarray(kps, dtype=np.float32)) if kps is not None else 1.0

            # Recorte com margem para a miniatura do painel.
            mw, mh = int(lw * 0.28), int(lh * 0.32)
            cx1, cy1 = max(0, x1 - mw), max(0, y1 - mh)
            cx2, cy2 = min(w_frame, x2 + mw), min(h_frame, y2 + mh)
            crop = frame[cy1:cy2, cx1:cx2].copy() if (cy2 > cy1 and cx2 > cx1) else None

            score, nitidez = qualidade_rosto(crop, lado, det_score, desvio)

            emb = getattr(f, "normed_embedding", None)
            if emb is None:
                emb = getattr(f, "embedding", None)
            emb = normalizar_vetor(emb) if emb is not None else None
            if emb is None or emb.size == 0:
                continue

            # no InsightFace gender == 1 -> Masculino, gender == 0 -> Feminino.
            genero = "Indeterminado"
            g = getattr(f, "gender", None)
            if g is not None:
                genero = "Masculino" if int(g) == 1 else "Feminino"

            resultado.append({
                "bbox": (x1, y1, x2, y2),
                "lado": lado,
                "crop": crop,
                "embedding": emb,
                "genero": genero,
                "idade": int(getattr(f, "age", 0) or 0),
                "det_score": det_score,
                "desvio_frontal": desvio,
                "qualidade": score,
                "nitidez": nitidez,
            })

        return resultado

class Track:
    """Uma pessoa acompanhada quadro a quadro enquanto esta em cena."""

    def __init__(self, track_id, bbox, ts):
        self.track_id = track_id
        self.bbox = bbox
        self.first_seen = ts
        self.last_seen = ts
        self.frames_bons = 0      
        self.frames_frontais = 0   

        self.pessoa_id = None
        self.nome = None
        self.genero = "Indeterminado"
        self.similaridade = 0.0
        self.status = "ANALISANDO"
        self.identidade_travada = False

        self.melhor_qualidade = -1.0
        self.melhor_crop = None
        self.embeddings = []            
        self.melhor_qualidade_frontal = -1.0
        self.melhor_crop_frontal = None
        self.embeddings_frontais = []   
        self.votos_genero = []
        self.evento_gravado = False
        self.template_aprendido = False
        self.motivo = ""          
        self.origem = "frontal"   
        self.match_votos = []     


    def acumular(self, det):
        emb, crop, q = det["embedding"], det["crop"], det["qualidade"]
        desvio, nitidez = det["desvio_frontal"], det["nitidez"]
        nitido_ok = nitidez >= CONFIG["NITIDEZ_MIN"]
        coleta_ok = desvio <= CONFIG["FRONTALIDADE_MAX"] and nitido_ok
        frontal_ok = desvio <= CONFIG["FRONTALIDADE_CADASTRO"] and nitido_ok

        if not nitido_ok:
            self.motivo = "desfocado"
        elif not coleta_ok:
            self.motivo = "de costas"
        elif not frontal_ok:
            self.motivo = "de perfil"
        elif q < CONFIG["QUALIDADE_MIN_CADASTRO"]:
            self.motivo = ("rosto distante" if det["lado"] < 34
                           else f"qualidade {q:.0f}")
        else:
            self.motivo = ""

        if emb is None:
            return

        if coleta_ok:
            self.frames_bons += 1
            self.embeddings.append((q, emb))
            self.embeddings.sort(key=lambda t: t[0], reverse=True)
            del self.embeddings[12:]

        if frontal_ok:
            self.frames_frontais += 1
            self.embeddings_frontais.append((q, emb))
            self.embeddings_frontais.sort(key=lambda t: t[0], reverse=True)
            del self.embeddings_frontais[12:]

        # Voto de sexo: so em frame de ALTA qualidade. O modelo de genero erra
        # com confianca em rosto pequeno, torto ou escuro -- e melhor nao votar
        # do que votar errado. Gate mais rigido que o de cadastro.
        if (det["genero"] != "Indeterminado"
                and det["lado"] >= CONFIG["GENERO_FRAME_LADO_MIN"]
                and desvio <= CONFIG["GENERO_FRAME_FRONTAL_MAX"]
                and nitidez >= CONFIG["GENERO_FRAME_NITIDEZ_MIN"]
                and det["det_score"] >= CONFIG["GENERO_FRAME_DET_MIN"]):
            self.votos_genero.append(det["genero"])
            del self.votos_genero[:-40]

        if crop is not None and crop.size > 0:
            if q > self.melhor_qualidade:
                self.melhor_qualidade = q
                self.melhor_crop = crop.copy()
            if frontal_ok and q > self.melhor_qualidade_frontal:
                self.melhor_qualidade_frontal = q
                self.melhor_crop_frontal = crop.copy()

    @staticmethod
    def _media(pares):
        if not pares:
            return None
        return normalizar_vetor(np.mean(np.vstack([e for _, e in pares[:6]]), axis=0))

    def embedding_consolidado(self):
        """Vetor para COMPARAR: media dos melhores frames, inclusive de perfil."""
        return self._media(self.embeddings) if self.embeddings else None

    def embedding_para_cadastro(self):
        """Vetor para CADASTRAR: apenas frames de frente."""
        return self._media(self.embeddings_frontais)

    def pode_decidir_match(self):
        """
        So confia num resultado de reconhecimento depois de juntar frames bons
        suficientes -- um embedding de 1-2 frames pequenos e ruido e da match
        aleatorio com qualquer pessoa da base.
        """
        return (self.frames_bons >= CONFIG["RECOG_FRAMES_MIN"]
                and self.melhor_qualidade >= CONFIG["RECOG_QUALIDADE_MIN"])

    def registrar_voto_match(self, pessoa_id, sim):
        self.match_votos.append((pessoa_id, float(sim)))
        del self.match_votos[:-12]

    def match_consistente(self, pessoa_id):
        """A MESMA pessoa foi o topo em frames suficientes, com media alta?"""
        recentes = self.match_votos[-CONFIG["RECOG_JANELA"]:]
        sims = [s for pid, s in recentes if pid == pessoa_id]
        return (len(sims) >= CONFIG["RECOG_CONSISTENCIA"]
                and sum(sims) / len(sims) >= CONFIG["SIM_RECONHECER"])

    def genero_por_votacao(self):
        """
        Exige quorum + maioria esmagadora. Sem isso, retorna Indeterminado
        (o sistema prefere 'nao_reconhecido' a chutar o sexo errado).
        """
        total = len(self.votos_genero)
        if total < CONFIG["GENERO_VOTOS_MIN"]:
            return "Indeterminado", 0.0
        m = self.votos_genero.count("Masculino")
        f = total - m
        vencedor, votos = ("Masculino", m) if m >= f else ("Feminino", f)
        proporcao = votos / float(total)
        if proporcao < CONFIG["GENERO_MAIORIA_MIN"]:
            return "Indeterminado", proporcao
        return vencedor, proporcao

    def pronto_para_cadastro(self):
        """Caminho ideal: evidencia frontal suficiente para uma referencia boa."""
        return (self.frames_frontais >= CONFIG["FRAMES_MIN_CADASTRO"]
                and self.melhor_qualidade_frontal >= CONFIG["QUALIDADE_MIN_CADASTRO"]
                and len(self.embeddings_frontais) >= 3)

    def pronto_para_cadastro_fallback(self, ts):
        """
        Caminho de garantia: a pessoa ficou tempo suficiente em cena e teve o
        rosto visivel varias vezes, mas nunca de frente. Registra assim mesmo.
        """
        return ((ts - self.first_seen) >= CONFIG["CADASTRO_FALLBACK_SEGUNDOS"]
                and self.frames_bons >= CONFIG["CADASTRO_FALLBACK_FRAMES"]
                and len(self.embeddings) >= 3)

    def melhor_crop_disponivel(self):
        """Recorte frontal se existir; senao o melhor recorte geral."""
        return self.melhor_crop_frontal if self.melhor_crop_frontal is not None \
            else self.melhor_crop

class SistemaReconhecimentoFacial:

    def __init__(self):
        self.db = DatabaseManager(DB_PATH)
        self.engine = FaceEngine()
        self.camera = RTSPCameraStream(CONFIG["RTSP_URL"])
        self.cache = ImageCache()

        self.tracks = {}
        self.proximo_track_id = 1
        self.contador_frames = 0
        self.deteccoes_cache = []

        self._enroll_lock = threading.Lock()
        self.galeria = []
        self.recarregar_galeria()

        # Card de comparacao: ESQUERDA = referencia, DIREITA = ultima passagem.
        self.destaque = None
        self.fps = 0.0
        self._fps_t0 = time.time()
        self._fps_n = 0

        # O painel nao pode consultar o banco a cada quadro: cache de 0.7s.
        self._painel_cache = None
        self._painel_t0 = 0.0

    def _dados_painel(self, forcar=False):
        agora = time.time()
        if forcar or self._painel_cache is None or (agora - self._painel_t0) > 0.7:
            self._painel_cache = (
                self.db.estatisticas(),
                self.db.historico_recente(CONFIG["HISTORICO_PAINEL"]),
            )
            self._painel_t0 = agora
        return self._painel_cache

    def recarregar_galeria(self):
        self.galeria = self.db.carregar_galeria()
        logging.info("Galeria: %d pessoas em memoria.", len(self.galeria))

    def _buscar_identidade(self, emb):
        """
        Compara com TODOS os templates de TODAS as pessoas (max por pessoa).
        Retorna (pessoa, similaridade). pessoa=None se a galeria esta vazia.
        """
        melhor, melhor_sim, _ = self._buscar_top2(emb)
        return melhor, melhor_sim

    def _buscar_top2(self, emb):
        """
        Retorna (melhor_pessoa, melhor_sim, sim_da_2a_pessoa_mais_parecida).
        A folga entre a 1a e a 2a diz se o match e inequivoco ou ambiguo.
        """
        ranking = []
        for pessoa in self.galeria:
            s = float(np.max(pessoa["templates"] @ emb))
            ranking.append((s, pessoa))
        if not ranking:
            return None, 0.0, 0.0
        ranking.sort(key=lambda t: t[0], reverse=True)
        melhor_sim, melhor = ranking[0]
        segundo_sim = ranking[1][0] if len(ranking) > 1 else 0.0
        return melhor, melhor_sim, segundo_sim

    def _match_confiavel(self, emb):
        """
        (pessoa, sim, sim2) so quando o match e SEGURO num frame isolado:
        similaridade >= SIM_RECONHECER e folga >= SIM_MARGEM_2A para a 2a
        pessoa. Senao (None, melhor_sim, sim2) -- serve para saber a zona cinza.
        """
        melhor, melhor_sim, segundo_sim = self._buscar_top2(emb)
        if melhor is None:
            return None, 0.0, 0.0
        if melhor_sim < CONFIG["SIM_RECONHECER"]:
            return None, melhor_sim, segundo_sim
        if (melhor_sim - segundo_sim) < CONFIG["SIM_MARGEM_2A"]:
            return None, melhor_sim, segundo_sim
        return melhor, melhor_sim, segundo_sim

    def _nome_para_genero(self, genero):
        if genero not in ("Masculino", "Feminino"):
            return CONFIG["NOME_INDEFINIDO"]
        pool = NOMES_MASCULINOS if genero == "Masculino" else NOMES_FEMININOS
        usados = self.db.nomes_em_uso()
        livres = [n for n in pool if n not in usados]
        if livres:
            return random.choice(livres)
        base = random.choice(pool)
        i = 2
        while f"{base} {i}" in usados:
            i += 1
        return f"{base} {i}"


    def _salvar_referencia(self, pessoa_id, crop):
        try:
            if crop is None or crop.size == 0:
                return ""
            pasta = os.path.join(REGISTRO_DIR, pessoa_id)
            os.makedirs(pasta, exist_ok=True)
            return salvar_imagem_rosto(os.path.join(pasta, "referencia.jpg"), crop)
        except Exception as e:
            logging.error("Erro ao salvar referencia de %s: %s", pessoa_id, e)
        return ""

    def _salvar_reconhecimento(self, pessoa_id, crop):
        try:
            if crop is None or crop.size == 0:
                return ""
            pasta = os.path.join(COMPARACOES_DIR, pessoa_id)
            os.makedirs(pasta, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            return salvar_imagem_rosto(os.path.join(pasta, f"reconh_{ts}.jpg"), crop)
        except Exception as e:
            logging.error("Erro ao salvar reconhecimento de %s: %s", pessoa_id, e)
        return ""


    def _associar(self, deteccoes, ts):
        assoc = {}
        usados = set()
        ordenadas = sorted(self.tracks.values(), key=lambda t: t.last_seen, reverse=True)

        for track in ordenadas:
            melhor_j, melhor_iou = -1, CONFIG["TRACK_IOU_MIN"]
            for j, d in enumerate(deteccoes):
                if j in usados:
                    continue
                v = iou(track.bbox, d["bbox"])
                if v > melhor_iou:
                    melhor_iou, melhor_j = v, j
            if melhor_j >= 0:
                usados.add(melhor_j)
                assoc[melhor_j] = track
                track.bbox = deteccoes[melhor_j]["bbox"]

        for j, d in enumerate(deteccoes):
            if j in usados:
                continue
            t = Track(self.proximo_track_id, d["bbox"], ts)
            self.proximo_track_id += 1
            self.tracks[t.track_id] = t
            assoc[j] = t
        return assoc

    def _limpar_tracks(self, ts):
        for tid in [t for t, tr in self.tracks.items()
                    if (ts - tr.last_seen) > CONFIG["TRACK_MAX_MISSING"]]:
            del self.tracks[tid]

    def _tentar_resolver_genero(self, track):
        """
        Da sexo e nome a quem foi cadastrado como 'nao_reconhecido' assim que a
        votacao atinge quorum. Um sexo ja definido nunca e alterado.
        """
        if not track.pessoa_id or track.genero != "Indeterminado":
            return
        genero, _ = track.genero_por_votacao()
        if genero == "Indeterminado":
            return
        nome = self._nome_para_genero(genero)
        if self.db.resolver_genero_indefinido(track.pessoa_id, genero, nome):
            track.genero, track.nome = genero, nome
            self.recarregar_galeria()
            if self.destaque and self.destaque["pessoa_id"] == track.pessoa_id:
                self.destaque["nome"], self.destaque["genero"] = nome, genero
            print(f"[SEXO DEFINIDO] {track.pessoa_id} - {nome} ({genero})")

    def _travar_reconhecido(self, track, pessoa, sim, emb):
        """Anexa o track a uma pessoa ja existente e trava a identidade."""
        track.pessoa_id = pessoa["id"]
        track.nome = pessoa["nome"]
        track.genero = pessoa["genero"]
        track.similaridade = sim
        track.status = "RECONHECIDO"
        track.identidade_travada = True
        self._tentar_resolver_genero(track)

        if (not track.template_aprendido
                and sim >= CONFIG["SIM_APRENDER"]
                and track.melhor_qualidade >= CONFIG["QUALIDADE_MIN_CADASTRO"]):
            self.db.adicionar_template(pessoa["id"], emb, track.melhor_qualidade)
            track.template_aprendido = True
            self.recarregar_galeria()

    def _cadastrar_novo(self, track, emb_cadastro, crop, qualidade, origem):
        if emb_cadastro is None:
            emb_cadastro = track.embedding_consolidado()
        if emb_cadastro is None:
            track.status = "ANALISANDO"
            return False

        with self._enroll_lock:
            self.recarregar_galeria()
            # Guarda anti-duplicacao: so anexa a um ID existente se o match for seguro.
            pessoa, sim, _ = self._match_confiavel(emb_cadastro)
            if pessoa is not None:
                self._travar_reconhecido(track, pessoa, sim, emb_cadastro)
                return True

            genero, _ = track.genero_por_votacao()
            nome = self._nome_para_genero(genero)
            novo_id = self.db.proximo_id()
            ref_crop = crop if crop is not None else track.melhor_crop
            caminho_ref = self._salvar_referencia(novo_id, ref_crop)

            if not self.db.cadastrar_pessoa(novo_id, nome, genero, emb_cadastro,
                                            caminho_ref, max(0.0, qualidade)):
                track.status = "ANALISANDO"
                return False
            self.recarregar_galeria()

        track.pessoa_id = novo_id
        track.nome = nome
        track.genero = genero
        track.similaridade = 1.0
        track.status = "NOVO"
        track.identidade_travada = True
        track.origem = origem
        if origem == "fallback":
            logging.info("CADASTRO FALLBACK (sem rosto de frente): %s - %s", novo_id, nome)
            print(f"[REGISTRO*]     {novo_id} - {nome} ({genero})"
                  f"  -- sem rosto de frente, qualidade reduzida")
        return True

    def _identificar(self, track, ts):
        if track.identidade_travada:
            self._tentar_resolver_genero(track)
            return

        emb = track.embedding_consolidado()
        if emb is None:
            track.status = "ANALISANDO"
            return

        # 1) Reconhecimento: exige evidencia suficiente e match consistente.
        if track.pode_decidir_match():
            pessoa, sim, sim2 = self._match_confiavel(emb)
            track.registrar_voto_match(pessoa["id"] if pessoa else None, sim)
            track.similaridade = sim
            if pessoa is not None and track.match_consistente(pessoa["id"]):
                self._travar_reconhecido(track, pessoa, sim, emb)
                return
            melhor_sim = sim
        else:
            _, melhor_sim, _ = self._buscar_top2(emb)
            track.similaridade = melhor_sim

        em_zona_cinza = melhor_sim >= CONFIG["SIM_ZONA_CINZA"]

        # 2) Cadastro ideal: rosto de frente com evidencia suficiente.
        if track.pronto_para_cadastro():
            self._cadastrar_novo(track, track.embedding_para_cadastro(),
                                 track.melhor_crop_frontal,
                                 track.melhor_qualidade_frontal, "frontal")
            return

        # 3) Fallback: nunca ficou de frente, mas ficou tempo suficiente em cena.
        if track.pronto_para_cadastro_fallback(ts):
            self._cadastrar_novo(track, emb, track.melhor_crop_disponivel(),
                                 track.melhor_qualidade, "fallback")
            return

        track.status = "CONFIRMANDO" if em_zona_cinza else "ANALISANDO"


    def _registrar_evento(self, track):
        if track.evento_gravado or not track.identidade_travada or not track.pessoa_id:
            return

        pessoa = self.db.obter_pessoa(track.pessoa_id)
        if pessoa is None:
            return

        primeira_vez = not self.db.tem_registro(track.pessoa_id)

        if primeira_vez:
            caminho_ref = pessoa["foto_referencia"]
            if not caminho_ref or not os.path.exists(caminho_ref):
                # A referencia e permanente: prefere sempre o recorte frontal.
                caminho_ref = self._salvar_referencia(
                    track.pessoa_id, track.melhor_crop_disponivel())
                self.db.definir_foto_referencia(track.pessoa_id, caminho_ref)
            self.db.registrar_evento(track.pessoa_id, "REGISTRO", caminho_ref, 1.0)
            track.evento_gravado = True
            self._atualizar_destaque(track.pessoa_id, caminho_ref, None, 1.0)
            print(f"[REGISTRO]      {track.pessoa_id} - {track.nome} ({track.genero})")
            return

        # Cooldown: nao registra a mesma pessoa varias vezes seguidas.
        ultima = pessoa["ultima_deteccao"]
        if ultima:
            try:
                dt = datetime.strptime(ultima, "%Y-%m-%d %H:%M:%S")
                if (datetime.now() - dt).total_seconds() < CONFIG["COOLDOWN_RECONHECIMENTO"]:
                    track.evento_gravado = True   # em cooldown: nada e gravado
                    self._atualizar_destaque_do_banco(track.pessoa_id, track.similaridade)
                    return
            except ValueError:
                pass

        caminho_rec = self._salvar_reconhecimento(track.pessoa_id, track.melhor_crop)
        self.db.registrar_evento(track.pessoa_id, "RECONHECIMENTO",
                                 caminho_rec, track.similaridade)
        track.evento_gravado = True
        self._atualizar_destaque(track.pessoa_id, pessoa["foto_referencia"],
                                 caminho_rec, track.similaridade)
        print(f"[RECONHECIDO]   {track.pessoa_id} - {track.nome} "
              f"({track.similaridade * 100:.0f}% de similaridade)")

    def _atualizar_destaque(self, pessoa_id, ref, rec, sim):
        pessoa = self.db.obter_pessoa(pessoa_id)
        if pessoa is None:
            return
        self._dados_painel(forcar=True)  
        agora = datetime.now().strftime("%d/%m %H:%M:%S")
        anterior = self.destaque or {}
        self.destaque = {
            "pessoa_id": pessoa_id,
            "nome": pessoa["nome"],
            "genero": pessoa["genero"],
            "ref_path": ref or pessoa["foto_referencia"],
            "ref_data": pessoa["data_cadastro"],
            "rec_path": rec,
            "rec_data": agora if rec else (
                anterior.get("rec_data") if anterior.get("pessoa_id") == pessoa_id else None),
            "similaridade": sim,
            "aparicoes": pessoa["total_aparicoes"],
        }

    def _atualizar_destaque_do_banco(self, pessoa_id, sim):
        pessoa = self.db.obter_pessoa(pessoa_id)
        if pessoa is None:
            return
        ultimo = self.db.ultimo_reconhecimento(pessoa_id)
        self.destaque = {
            "pessoa_id": pessoa_id,
            "nome": pessoa["nome"],
            "genero": pessoa["genero"],
            "ref_path": pessoa["foto_referencia"],
            "ref_data": pessoa["data_cadastro"],
            "rec_path": ultimo[1] if ultimo else None,
            "rec_data": (datetime.strptime(ultimo[0], "%Y-%m-%d %H:%M:%S")
                         .strftime("%d/%m %H:%M:%S")) if ultimo else None,
            "similaridade": sim,
            "aparicoes": pessoa["total_aparicoes"],
        }

    def _processar_frame(self, frame):
        ts = time.time()
        deteccoes = self.engine.analisar(frame)
        assoc = self._associar(deteccoes, ts)

        for j, d in enumerate(deteccoes):
            track = assoc[j]
            track.last_seen = ts
            track.acumular(d)
            self._identificar(track, ts)
            self._registrar_evento(track)
            d["track"] = track

        self._limpar_tracks(ts)
        return deteccoes


    def _desenhar_rostos(self, canvas, deteccoes, escala_x, escala_y):
        for d in deteccoes:
            track = d.get("track")
            if track is None:
                continue
            x1, y1, x2, y2 = d["bbox"]
            X1, Y1 = int(x1 * escala_x), int(y1 * escala_y)
            X2, Y2 = int(x2 * escala_x), int(y2 * escala_y)

            if track.status in ("RECONHECIDO", "NOVO") and track.pessoa_id:
                # Pessoa ja registrada em cena: SEMPRE caixa verde com o ID fixo
                # e o nome. O ID vem do banco e nunca muda.
                cor = CONFIG["C_VERDE"]
                rotulo = f"{track.pessoa_id} - {track.nome}"
            elif track.status == "CONFIRMANDO":
                cor = CONFIG["C_AMARELO"]
                rotulo = "Confirmando..."
            else:
                # Diz POR QUE ainda nao cadastrou, em vez de um "Analisando..." mudo.
                cor = CONFIG["C_MUTED"]
                rotulo = f"Analisando ({track.motivo})" if track.motivo else "Analisando..."

            retangulo_arredondado(canvas, (X1, Y1), (X2, Y2), cor, 2, 10)

            (tw, th), _ = cv2.getTextSize(sem_acento(rotulo),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            tag_y = max(th + 12, Y1 - 6)
            retangulo_arredondado(canvas, (X1, tag_y - th - 10),
                                  (X1 + tw + 14, tag_y + 4), (18, 16, 22), -1, 5)
            retangulo_arredondado(canvas, (X1, tag_y - th - 10),
                                  (X1 + tw + 14, tag_y + 4), cor, 1, 5)
            texto(canvas, rotulo, (X1 + 7, tag_y - 3), 0.45, cor, 1)

    @staticmethod
    def _texto_centrado(canvas, msg, cx, y, escala, cor, largura_max):
        """Escreve msg centrada em cx, reduzindo a escala ate caber em largura_max."""
        msg = sem_acento(msg)
        s = escala
        (tw, _), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, s, 1)
        while tw > largura_max and s > 0.2:
            s -= 0.02
            (tw, _), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, s, 1)
        cv2.putText(canvas, msg, (int(cx - tw / 2), y), cv2.FONT_HERSHEY_SIMPLEX,
                    s, cor, 1, cv2.LINE_AA)

    def _desenhar_caixa_foto(self, canvas, x, y, lado, caminho, cor_borda, legenda,
                             cor_legenda, placeholder=("AGUARDANDO", "")):
        retangulo_arredondado(canvas, (x - 2, y - 2), (x + lado + 2, y + lado + 2),
                              CONFIG["C_BG"], -1, 8)
        img = self.cache.get(caminho) if caminho else None
        interno = lado - 6
        if img is not None:
            thumb = letterbox(img, interno, interno)
            canvas[y + 3:y + 3 + interno, x + 3:x + 3 + interno] = thumb
        else:
            retangulo_arredondado(canvas, (x + 3, y + 3),
                                  (x + lado - 3, y + lado - 3), (34, 30, 40), -1, 6)
            linhas = [l for l in placeholder if l]
            base = y + lado // 2 - (len(linhas) - 1) * 8
            for k, linha in enumerate(linhas):
                self._texto_centrado(canvas, linha, x + lado / 2, base + k * 16,
                                     0.30, CONFIG["C_MUTED"], lado - 12)
        retangulo_arredondado(canvas, (x, y), (x + lado, y + lado), cor_borda, 2, 8)
        self._texto_centrado(canvas, legenda, x + lado / 2, y + lado + 15,
                             0.3, cor_legenda, lado + 20)

    def _desenhar_painel(self, canvas, px, altura, online):
        largura_total = canvas.shape[1]
        cv2.rectangle(canvas, (px, 0), (largura_total, altura), CONFIG["C_BG"], -1)
        cv2.line(canvas, (px, 0), (px, altura), (48, 44, 58), 1)

        # ---- cabecalho ----
        texto(canvas, "C.I.S FACIAL", (px + 20, 38), 0.72, CONFIG["C_PRIMARY"], 2)
        texto(canvas, "SISTEMA ATIVO" if online else "SISTEMA OFFLINE", (px + 21, 56),
              0.31, CONFIG["C_MUTED"] if online else CONFIG["C_VERMELHO"], 1)

        pill_bg = (32, 62, 38) if online else (32, 32, 62)
        pill_fg = CONFIG["C_VERDE"] if online else CONFIG["C_VERMELHO"]
        retangulo_arredondado(canvas, (px + 272, 20), (px + 366, 48), pill_bg, -1, 7)
        texto(canvas, "ONLINE" if online else "OFFLINE", (px + 292, 39), 0.38, pill_fg, 1)

        # ---- estatisticas ----
        retangulo_arredondado(canvas, (px + 14, 72), (px + 366, 142), CONFIG["C_CARD"], -1, 8)
        retangulo_arredondado(canvas, (px + 14, 72), (px + 366, 142), CONFIG["C_BORDA"], 1, 8)
        (cadastrados, registros, reconhecimentos, hoje), eventos = self._dados_painel()
        texto(canvas, "BANCO DE DADOS", (px + 28, 93), 0.36, CONFIG["C_MUTED"], 1)
        texto(canvas, f"{self.fps:.0f} fps", (px + 310, 93), 0.32, CONFIG["C_MUTED"], 1)

        colunas = [
            (px + 36, str(cadastrados), "Pessoas", CONFIG["C_TEXTO"]),
            (px + 122, str(registros), "Registros", CONFIG["C_VERDE"]),
            (px + 208, str(reconhecimentos), "Reconhec.", CONFIG["C_CIANO"]),
            (px + 296, str(hoje), "Hoje", CONFIG["C_PRIMARY"]),
        ]
        for cx, valor, rotulo, cor in colunas:
            texto(canvas, valor, (cx, 122), 0.62, cor, 2)
            texto(canvas, rotulo, (cx - 4, 136), 0.29, CONFIG["C_MUTED"], 1)

        # ---- comparacao lado a lado ----
        retangulo_arredondado(canvas, (px + 14, 152), (px + 366, 372), CONFIG["C_CARD"], -1, 8)
        retangulo_arredondado(canvas, (px + 14, 152), (px + 366, 372), CONFIG["C_BORDA"], 1, 8)
        texto(canvas, "COMPARACAO FACIAL", (px + 28, 174), 0.38, CONFIG["C_PRIMARY"], 1)

        d = self.destaque
        lado, y_box = 104, 188
        x_esq, x_dir = px + 74, px + 202

        # ESQUERDA = REGISTRO (1a imagem capturada).
        # DIREITA  = RECONHECIMENTO (a pessoa passa de novo e as duas sao comparadas).
        ph_esq = ("AGUARDANDO", "REGISTRO")
        ph_dir = ("AGUARDANDO", "RECONHECIMENTO")

        if not d:
            self._desenhar_caixa_foto(canvas, x_esq, y_box, lado, None,
                                      (58, 54, 70), "REGISTRO", CONFIG["C_MUTED"], ph_esq)
            self._desenhar_caixa_foto(canvas, x_dir, y_box, lado, None,
                                      (58, 54, 70), "RECONHECIMENTO", CONFIG["C_MUTED"], ph_dir)
            texto(canvas, "Aguardando o primeiro registro...",
                  (px + 66, 340), 0.35, CONFIG["C_MUTED"], 1)
        else:
            # ESQUERDA: sempre a PRIMEIRA imagem registrada da pessoa.
            self._desenhar_caixa_foto(canvas, x_esq, y_box, lado, d["ref_path"],
                                      (92, 86, 108), "REGISTRO (1a)", CONFIG["C_MUTED"], ph_esq)
            # DIREITA: a captura da passagem em que a pessoa foi reconhecida.
            cor_dir = CONFIG["C_VERDE"] if d["rec_path"] else (58, 54, 70)
            self._desenhar_caixa_foto(canvas, x_dir, y_box, lado, d["rec_path"],
                                      cor_dir, "RECONHECIMENTO",
                                      CONFIG["C_VERDE"] if d["rec_path"] else CONFIG["C_MUTED"],
                                      ph_dir)

            indefinido = d["nome"] == CONFIG["NOME_INDEFINIDO"]
            cor_nome = CONFIG["C_AMARELO"] if indefinido else CONFIG["C_TEXTO"]
            rotulo = texto_limitado(f"{d['pessoa_id']} - {d['nome']}", 326, 0.42, 1)
            texto(canvas, rotulo, (px + 28, 328), 0.42, cor_nome, 1)

            sexo = {"Masculino": "Masculino", "Feminino": "Feminino"}.get(
                d["genero"], "Sexo indefinido")
            texto(canvas, f"{sexo}  |  {d['aparicoes']} aparicoes", (px + 28, 346),
                  0.32, CONFIG["C_MUTED"], 1)

            if d["rec_path"]:
                info = f"Similaridade: {d['similaridade'] * 100:.0f}%   {d['rec_data']}"
                texto(canvas, info, (px + 28, 363), 0.32, CONFIG["C_VERDE"], 1)
            else:
                texto(canvas, f"Cadastrado em {d['ref_data']}", (px + 28, 363),
                      0.32, CONFIG["C_MUTED"], 1)

        y0, y1 = 382, altura - 12
        retangulo_arredondado(canvas, (px + 14, y0), (px + 366, y1), CONFIG["C_CARD"], -1, 8)
        retangulo_arredondado(canvas, (px + 14, y0), (px + 366, y1), CONFIG["C_BORDA"], 1, 8)
        texto(canvas, "HISTORICO DE REGISTROS", (px + 28, y0 + 22), 0.38, CONFIG["C_PRIMARY"], 1)

        y = y0 + 46
        if not eventos:
            texto(canvas, "Nenhum registro ainda.", (px + 28, y + 6), 0.32, CONFIG["C_MUTED"], 1)
        for data_hora, tipo, pid, nome, _genero, sim in eventos:
            if y > y1 - 8:
                break
            novo = tipo == "REGISTRO"
            cor = CONFIG["C_VERDE"] if novo else CONFIG["C_CIANO"]
            try:
                hora = datetime.strptime(data_hora, "%Y-%m-%d %H:%M:%S").strftime("%d/%m %H:%M")
            except ValueError:
                hora = data_hora[:11]
            cv2.circle(canvas, (px + 30, y - 4), 3, cor, -1, cv2.LINE_AA)
            texto(canvas, hora, (px + 40, y), 0.3, CONFIG["C_MUTED"], 1)
            texto(canvas, pid, (px + 118, y), 0.3, cor, 1)
            texto(canvas, texto_limitado(nome, 148, 0.3, 1), (px + 176, y), 0.3,
                  CONFIG["C_TEXTO"], 1)
            texto(canvas, "1a" if novo else f"{sim * 100:.0f}%", (px + 330, y), 0.3, cor, 1)
            y += 19

    def renderizar(self, frame, deteccoes, online):
        disp_w, disp_h = CONFIG["DISPLAY_WIDTH"], CONFIG["DISPLAY_HEIGHT"]
        panel_w = CONFIG["PANEL_WIDTH"]
        canvas = np.zeros((disp_h, disp_w + panel_w, 3), dtype=np.uint8)

        if frame is not None and frame.size > 0:
            h_orig, w_orig = frame.shape[:2]
            video = cv2.resize(frame, (disp_w, disp_h))
            escala_x, escala_y = disp_w / w_orig, disp_h / h_orig
        else:
            video = np.full((disp_h, disp_w, 3), (18, 16, 22), dtype=np.uint8)
            escala_x = escala_y = 1.0

        if online:
            self._desenhar_rostos(video, deteccoes, escala_x, escala_y)
        else:
            texto(video, "SINAL PERDIDO - RECONECTANDO",
                  (disp_w // 2 - 180, disp_h // 2), 0.7, CONFIG["C_VERMELHO"], 2)

        canvas[0:disp_h, 0:disp_w] = video
        self._desenhar_painel(canvas, disp_w, disp_h, online)
        return canvas


    def exibir_historico(self):
        linhas = self.db.historico_completo()
        print("\n" + "=" * 96)
        print("HISTORICO COMPLETO - C.I.S FACIAL v5.0")
        print("=" * 96)
        if not linhas:
            print("\nNenhum registro encontrado.\n")
            _pausar()
            return

        cadastrados, registros, reconhecimentos, hoje = self.db.estatisticas()
        print(f"\nPessoas: {cadastrados}   Registros: {registros}   "
              f"Reconhecimentos: {reconhecimentos}   Distintas hoje: {hoje}\n")
        print(f"{'DATA/HORA':<21}{'TIPO':<17}{'ID':<9}{'NOME':<26}{'SEXO':<14}{'SIM.':>6}")
        print("-" * 96)
        for data_hora, tipo, pid, nome, genero, sim, _img in linhas:
            marca = "REGISTRO (1a)" if tipo == "REGISTRO" else "RECONHECIMENTO"
            print(f"{data_hora:<21}{marca:<17}{pid:<9}{nome:<26}"
                  f"{(genero or 'Indeterminado'):<14}{sim * 100:>5.0f}%")
        print("-" * 96)
        print(f"Total de eventos: {len(linhas)}")
        print("=" * 96)
        _pausar()

    def exportar_csv(self):
        destino = os.path.join(
            BASE_DIR, f"historico_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        linhas = self.db.historico_completo()
        try:
            with open(destino, "w", newline="", encoding="utf-8-sig") as fp:
                w = csv.writer(fp, delimiter=";")
                w.writerow(["Data/Hora", "Tipo", "ID", "Nome", "Sexo",
                            "Similaridade (%)", "Imagem"])
                for data_hora, tipo, pid, nome, genero, sim, img in linhas:
                    w.writerow([data_hora, tipo, pid, nome, genero or "Indeterminado",
                                f"{sim * 100:.1f}", img or ""])
            print(f"\n[OK] Historico exportado: {destino}  ({len(linhas)} eventos)\n")
            logging.info("Historico exportado para %s", destino)
        except Exception as e:
            print(f"\n[ERRO] Falha ao exportar CSV: {e}\n")


    def run(self):
        if not self.engine.disponivel:
            print("\nO sistema nao pode iniciar sem o InsightFace.")
            print("  pip install insightface onnxruntime\n")
            return

        self.camera.start()
        cadastrados, registros, reconhecimentos, hoje = self.db.estatisticas()

        print("\n" + "=" * 70)
        print("  C.I.S FACIAL v5.0 PRO")
        print("=" * 70)
        print(f"  Motor            : InsightFace {self.engine.nome_modelo}")
        print(f"  Pessoas na base  : {cadastrados}")
        print(f"  Registros        : {registros}   Reconhecimentos: {reconhecimentos}")
        print(f"  Limiar de match  : {CONFIG['SIM_RECONHECER']:.2f}  "
              f"folga 2a pessoa {CONFIG['SIM_MARGEM_2A']:.2f}  "
              f"consistencia {CONFIG['RECOG_CONSISTENCIA']}/{CONFIG['RECOG_JANELA']}")
        print(f"  Cooldown         : {CONFIG['COOLDOWN_RECONHECIMENTO']}s por pessoa")
        print("=" * 70)
        print("  [H] historico   [E] exportar CSV   [R] recarregar   [Q] sair\n")

        janela = "C.I.S Facial v5.0 PRO"
        cv2.namedWindow(janela, cv2.WINDOW_AUTOSIZE)

        try:
            while True:
                conectado, frame = self.camera.read()

                if not conectado or frame is None:
                    cv2.imshow(janela, self.renderizar(None, [], False))
                    if (cv2.waitKey(80) & 0xFF) in (ord("q"), ord("Q"), 27):
                        break
                    continue

                self.contador_frames += 1
                if self.contador_frames % CONFIG["PROCESS_EVERY_N_FRAMES"] == 0:
                    self.deteccoes_cache = self._processar_frame(frame)

                cv2.imshow(janela, self.renderizar(frame, self.deteccoes_cache, True))

                self._fps_n += 1
                agora = time.time()
                if agora - self._fps_t0 >= 1.0:
                    self.fps = self._fps_n / (agora - self._fps_t0)
                    self._fps_n, self._fps_t0 = 0, agora

                tecla = cv2.waitKey(1) & 0xFF
                if tecla in (ord("q"), ord("Q"), 27):
                    break
                if tecla in (ord("h"), ord("H")):
                    cv2.destroyWindow(janela)
                    self.exibir_historico()
                    cv2.namedWindow(janela, cv2.WINDOW_AUTOSIZE)
                elif tecla in (ord("e"), ord("E")):
                    self.exportar_csv()
                elif tecla in (ord("r"), ord("R")):
                    self.recarregar_galeria()
                    print(f"[OK] Galeria recarregada: {len(self.galeria)} pessoas.")

        except KeyboardInterrupt:
            print("\nEncerrado pelo usuario.")
        finally:
            self.camera.stop()
            cv2.destroyAllWindows()
            print("Sistema finalizado.\n")


def diagnosticar(segundos=40):
    """
    Amostra a cena real e mostra quantos rostos passam em cada filtro, para
    ajustar o CONFIG com dados em vez de chute.
    """
    engine = FaceEngine()
    if not engine.disponivel:
        return

    cam = RTSPCameraStream(CONFIG["RTSP_URL"]).start()
    print("Conectando na camera...")
    inicio = time.time()
    while time.time() - inicio < 15:
        ok, frame = cam.read()
        if ok and frame is not None:
            break
        time.sleep(0.4)

    ok, frame = cam.read()
    if not ok or frame is None:
        print("CAMERA INDISPONIVEL. Verifique a rede e a URL RTSP em CONFIG.")
        cam.stop()
        return

    print(f"Stream: {frame.shape[1]}x{frame.shape[0]}")
    print(f"Amostrando por {segundos}s -- peca para alguem passar pela camera.\n")

    amostras, n_frames = [], 0
    inicio = time.time()
    while time.time() - inicio < segundos:
        ok, frame = cam.read()
        if not ok or frame is None:
            time.sleep(0.1)
            continue
        n_frames += 1
        h, w = frame.shape[:2]
        try:
            faces = engine.app.get(frame)
        except Exception as e:
            print(f"Erro na analise: {e}")
            break
        for f in faces:
            x1, y1, x2, y2 = [int(v) for v in f.bbox]
            lado = min(x2 - x1, y2 - y1)
            kps = getattr(f, "kps", None)
            desvio = frontalidade(np.asarray(kps, dtype=np.float32)) if kps is not None else 1.0
            crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
            q, nitidez = qualidade_rosto(crop, lado, float(f.det_score), desvio)
            amostras.append({"lado": lado, "det": float(f.det_score),
                             "frontal": desvio, "nitidez": nitidez, "qual": q})
        time.sleep(0.05)
    cam.stop()

    print(f"{n_frames} frames lidos, {len(amostras)} rostos detectados.")
    if not amostras:
        print("\nNenhum rosto apareceu na amostragem. Rode de novo com alguem em cena.")
        return

    def linha(nome, chave, fmt="{:.2f}"):
        a = np.array([x[chave] for x in amostras], dtype=float)
        print(f"  {nome:<14} min={fmt.format(a.min())}  mediana={fmt.format(np.median(a))}"
              f"  max={fmt.format(a.max())}")

    print("\n=== O QUE A CAMERA ENTREGA ===")
    linha("lado (px)", "lado", "{:.0f}")
    linha("det_score", "det")
    linha("frontalidade", "frontal")
    linha("nitidez", "nitidez", "{:.0f}")
    linha("qualidade", "qual", "{:.0f}")

    total = len(amostras)
    filtros = [
        ("tamanho >= %d px" % CONFIG["MIN_FACE_SIZE"],
         lambda a: a["lado"] >= CONFIG["MIN_FACE_SIZE"]),
        ("det_score >= %.2f" % CONFIG["DET_SCORE_MIN"],
         lambda a: a["det"] >= CONFIG["DET_SCORE_MIN"]),
        ("nitidez >= %.0f" % CONFIG["NITIDEZ_MIN"],
         lambda a: a["nitidez"] >= CONFIG["NITIDEZ_MIN"]),
        ("pose p/ comparar <= %.2f" % CONFIG["FRONTALIDADE_MAX"],
         lambda a: a["frontal"] <= CONFIG["FRONTALIDADE_MAX"]),
        ("pose p/ cadastrar <= %.2f" % CONFIG["FRONTALIDADE_CADASTRO"],
         lambda a: a["frontal"] <= CONFIG["FRONTALIDADE_CADASTRO"]),
        ("qualidade >= %.0f" % CONFIG["QUALIDADE_MIN_CADASTRO"],
         lambda a: a["qual"] >= CONFIG["QUALIDADE_MIN_CADASTRO"]),
    ]
    print("\n=== QUANTOS PASSAM EM CADA FILTRO ===")
    for nome, cond in filtros:
        n = sum(1 for a in amostras if cond(a))
        print(f"  {nome:<26} {n:>4}/{total}  ({n / total * 100:>3.0f}%)")

    aptos = sum(1 for a in amostras if all(c(a) for _, c in filtros))
    comparaveis = sum(1 for a in amostras
                      if a["lado"] >= CONFIG["MIN_FACE_SIZE"]
                      and a["det"] >= CONFIG["DET_SCORE_MIN"]
                      and a["nitidez"] >= CONFIG["NITIDEZ_MIN"]
                      and a["frontal"] <= CONFIG["FRONTALIDADE_MAX"])
    grau_genero = sum(
        1 for a in amostras
        if a["lado"] >= CONFIG["GENERO_FRAME_LADO_MIN"]
        and a["frontal"] <= CONFIG["GENERO_FRAME_FRONTAL_MAX"]
        and a["nitidez"] >= CONFIG["GENERO_FRAME_NITIDEZ_MIN"]
        and a["det"] >= CONFIG["GENERO_FRAME_DET_MIN"]
    )
    print(f"\n  SERVEM PARA RECONHECER     {comparaveis:>4}/{total}"
          f"  ({comparaveis / total * 100:>3.0f}%)")
    print(f"  APTOS A CADASTRO NOVO      {aptos:>4}/{total}"
          f"  ({aptos / total * 100:>3.0f}%)")
    print(f"  BONS PARA DEFINIR O SEXO   {grau_genero:>4}/{total}"
          f"  ({grau_genero / total * 100:>3.0f}%)  "
          f"[precisa de {CONFIG['GENERO_VOTOS_MIN']} para cravar o sexo]")

    print("\n=== LEITURA ===")
    if aptos == 0:
        gargalo = min(filtros, key=lambda f: sum(1 for a in amostras if f[1](a)))
        print(f"  Nenhum rosto apto. O filtro que mais barra e: {gargalo[0]}")
        mediana_lado = np.median([a["lado"] for a in amostras])
        mediana_front = np.median([a["frontal"] for a in amostras])
        if mediana_lado < 24:
            print(f"  Os rostos chegam com ~{mediana_lado:.0f}px. Abaixo de ~18px o modelo")
            print("  nao distingue pessoas. Aproxime a camera, use zoom optico ou")
            print("  aumente a resolucao do stream (subtype=0 e o canal principal).")
        if mediana_front > CONFIG["FRONTALIDADE_CADASTRO"]:
            print(f"  Frontalidade mediana {mediana_front:.2f} (frontal e ~0.21): as pessoas")
            print("  passam de lado. O sistema ate reconhece quem ja esta cadastrado,")
            print("  mas so cria um cadastro novo com o rosto de frente. Reposicione")
            print("  a camera para a linha de passagem vir na direcao dela.")
    else:
        print("  Os limiares atuais estao adequados para esta cena.")


def limpar_pastas(aplicar=False):
    """
    Remove pastas obsoletas e subpastas vazias de database/. Nunca toca numa
    pasta que contenha imagem referenciada pelo banco.
    """
    db = DatabaseManager(DB_PATH)
    protegidos = {os.path.abspath(p) for p in db.caminhos_de_imagem() if p}
    protegidas = {os.path.dirname(p) for p in protegidos}

    alvos = []

    for nome in PASTAS_OBSOLETAS:
        caminho = os.path.join(BASE_DIR, nome)
        if not os.path.isdir(caminho):
            continue
        imagens = [
            os.path.abspath(os.path.join(raiz, f))
            for raiz, _, arquivos in os.walk(caminho)
            for f in arquivos if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        if not imagens or not (set(imagens) & protegidos):
            alvos.append((caminho, "pasta obsoleta sem imagens em uso"))

    for pai in (REGISTRO_DIR, COMPARACOES_DIR):
        if not os.path.isdir(pai):
            continue
        for nome in os.listdir(pai):
            sub = os.path.join(pai, nome)
            if not os.path.isdir(sub) or os.path.abspath(sub) in protegidas:
                continue
            tem_imagem = any(
                f.lower().endswith((".jpg", ".jpeg", ".png"))
                for _, _, arquivos in os.walk(sub) for f in arquivos
            )
            if not tem_imagem:
                alvos.append((sub, "pasta de pessoa sem imagem"))

    for raiz, dirs, arquivos in os.walk(BASE_DIR, topdown=False):
        if os.path.abspath(raiz) in protegidas or raiz == BASE_DIR:
            continue
        if not arquivos and not dirs:
            alvos.append((raiz, "pasta vazia"))

    vistos, unicos = set(), []
    for caminho, motivo in alvos:
        chave = os.path.abspath(caminho)
        if chave not in vistos:
            vistos.add(chave)
            unicos.append((caminho, motivo))

    if not unicos:
        print("Nenhuma pasta descartavel encontrada. Estrutura ja esta limpa.")
        return

    print(f"\n{'REMOVIDAS' if aplicar else 'PASTAS DESCARTAVEIS (simulacao)'}: "
          f"{len(unicos)}\n")
    for caminho, motivo in unicos:
        print(f"  - {caminho}   ({motivo})")
        if aplicar:
            try:
                shutil.rmtree(caminho, ignore_errors=True)
            except Exception as e:
                print(f"      [ERRO] {e}")

    if aplicar:
        for p in PASTAS_ATIVAS:
            os.makedirs(p, exist_ok=True)
        print("\nEstrutura final: database/{registro, comparacoes, logs, system.db}\n")
    else:
        print("\nExecute com --limpar-pastas --aplicar para remover de fato.\n")


def corrigir_pessoa(pessoa_id, sexo):
    mapa = {"f": "Feminino", "feminino": "Feminino", "mulher": "Feminino",
            "m": "Masculino", "masculino": "Masculino", "homem": "Masculino",
            "?": "Indeterminado", "indefinido": "Indeterminado"}
    genero = mapa.get(str(sexo).strip().lower())
    if genero is None:
        print(f"Sexo invalido: '{sexo}'. Use F, M ou ?.")
        return

    db = DatabaseManager(DB_PATH)
    pessoa = db.obter_pessoa(pessoa_id)
    if pessoa is None:
        print(f"Pessoa {pessoa_id} nao encontrada.")
        return

    if genero == "Masculino":
        nome = random.choice(NOMES_MASCULINOS)
    elif genero == "Feminino":
        nome = random.choice(NOMES_FEMININOS)
    else:
        nome = CONFIG["NOME_INDEFINIDO"]

    usados = db.nomes_em_uso() - {pessoa["nome"]}
    tentativas = 0
    while nome in usados and tentativas < 50:
        nome = (random.choice(NOMES_MASCULINOS) if genero == "Masculino"
                else random.choice(NOMES_FEMININOS) if genero == "Feminino"
                else CONFIG["NOME_INDEFINIDO"])
        tentativas += 1

    if db.forcar_identidade(pessoa_id, genero, nome):
        print(f"[OK] {pessoa_id}: {pessoa['nome']} ({pessoa['genero']})  ->  "
              f"{nome} ({genero})")
    else:
        print(f"Nao foi possivel atualizar {pessoa_id}.")

def main():
    parser = argparse.ArgumentParser(description="C.I.S Facial v5.0 PRO")
    parser.add_argument("--historico", action="store_true",
                        help="mostra o historico completo e sai")
    parser.add_argument("--exportar", action="store_true",
                        help="exporta o historico em CSV e sai")
    parser.add_argument("--limpar-pastas", action="store_true",
                        help="lista pastas descartaveis (use com --aplicar para remover)")
    parser.add_argument("--aplicar", action="store_true",
                        help="confirma a remocao em --limpar-pastas")
    parser.add_argument("--diagnostico", action="store_true",
                        help="mede a cena da camera e mostra qual filtro esta barrando")
    parser.add_argument("--corrigir", nargs=2, metavar=("ID", "SEXO"),
                        help="corrige o sexo de uma pessoa: --corrigir ID_004 F "
                             "(F=Feminino, M=Masculino, ?=nao_reconhecido)")
    args = parser.parse_args()

    if args.corrigir:
        corrigir_pessoa(args.corrigir[0], args.corrigir[1])
        return

    if args.limpar_pastas:
        limpar_pastas(aplicar=args.aplicar)
        return

    if args.diagnostico:
        diagnosticar()
        return

    if args.historico or args.exportar:
        sistema = SistemaReconhecimentoFacial.__new__(SistemaReconhecimentoFacial)
        sistema.db = DatabaseManager(DB_PATH)
        if args.historico:
            SistemaReconhecimentoFacial.exibir_historico(sistema)
        if args.exportar:
            SistemaReconhecimentoFacial.exportar_csv(sistema)
        return

    SistemaReconhecimentoFacial().run()


if __name__ == "__main__":
    main()
