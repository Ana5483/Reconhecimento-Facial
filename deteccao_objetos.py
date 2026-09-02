import cv2
import json
import os
import threading
import time
import numpy as np
from datetime import datetime
from collections import deque
from ultralytics import YOLO

URL_RTSP = "rtsp://admin:%40nvd%401234%3F@192.168.3.28:554/cam/realmonitor?channel=1&subtype=0"
ARQUIVO_JSON = "object_database.json"

CLASSES_ALVO = {
    0: "pessoa",
    1: "bicicleta",
    2: "carro",
    3: "moto",
    7: "caminhao",
    15: "gato",
    16: "cachorro",
}

CORES_CLASSES = {
    "pessoa": (0, 255, 0),      
    "bicicleta": (255, 0, 255),  
    "carro": (255, 140, 0),      
    "moto": (0, 165, 255),       
    "caminhao": (0, 0, 255),     
    "gato": (203, 192, 255),     
    "cachorro": (0, 255, 255),  
}

LIMIAR_CONFIANCA = 0.55          
PROCESSAR_A_CADA_N_FRAMES = 3    
LARGURA_PAINEL_LATERAL = 320
MAX_HISTORICO_LOG = 6

TEMPO_COOLDOWN_REGISTRO = {
    "pessoa": 15.0,     
    "carro": 20.0,      
    "moto": 10.0,
    "bicicleta": 10.0,
    "caminhao": 25.0,
    "gato": 30.0,
    "cachorro": 30.0
}

ultimo_registro_por_classe = {classe: 0.0 for classe in CLASSES_ALVO.values()}

banco_dados = {"eventos": []}
if os.path.exists(ARQUIVO_JSON):
    try:
        with open(ARQUIVO_JSON, "r", encoding="utf-8") as f:
            conteudo = json.load(f)
            if isinstance(conteudo, dict) and "eventos" in conteudo:
                banco_dados = conteudo
    except Exception:
        pass

ultimas_posicoes_objetos = {classe: [] for classe in CLASSES_ALVO.values()}
historico_eventos = deque(maxlen=MAX_HISTORICO_LOG)
estatisticas_classes = {classe: 0 for classe in CLASSES_ALVO.values()}

for evento in banco_dados.get("eventos", []):
    obj = evento.get("objeto")
    if obj in estatisticas_classes:
        estatisticas_classes[obj] += 1

def calcular_iou(boxA, boxB):
    """Calcula a sobreposição entre duas caixas para saber se é o mesmo objeto parado"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

def salvar_json():
    try:
        with open(ARQUIVO_JSON, "w", encoding="utf-8") as f:
            json.dump(banco_dados, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"[ERRO] Falha ao salvar o JSON: {e}")

def registrar_deteccao(classe: str, confianca: float, bbox: list):
    agora = datetime.now()
    tempo_atual_seg = time.time()
    ultimo_tempo_registro = ultimo_registro_por_classe.get(classe, 0.0)
    cooldown_necessario = TEMPO_COOLDOWN_REGISTRO.get(classe, 15.0)
    
    if tempo_atual_seg - ultimo_tempo_registro < cooldown_necessario:
        return  

    objeto_parado = False
    posicoes_atualizadas = []
    
    for antiga_bbox, antigo_tempo in ultimas_posicoes_objetos[classe]:
        if tempo_atual_seg - antigo_tempo < 300:  
            posicoes_atualizadas.append((antiga_bbox, antigo_tempo))
            if calcular_iou(bbox, antiga_bbox) > 0.70:
                objeto_parado = True
                
    if objeto_parado:
        return 

    posicoes_atualizadas.append((bbox, tempo_atual_seg))
    ultimas_posicoes_objetos[classe] = posicoes_atualizadas
    ultimo_registro_por_classe[classe] = tempo_atual_seg  

    estatisticas_classes[classe] += 1

    evento = {
        "timestamp": agora.strftime("%Y-%m-%d %H:%M:%S"),
        "objeto": classe,
        "confianca": round(float(confianca), 2)
    }
    
    if "eventos" not in banco_dados:
        banco_dados["eventos"] = []
        
    banco_dados["eventos"].append(evento)
    salvar_json()
    
    historico_eventos.append({
        "hora": agora.strftime("%H:%M:%S"),
        "objeto": classe,
        "confianca": confianca
    })
    print(f"[REGISTRO] {evento['timestamp']} - {classe.upper()} ({evento['confianca']*100:.0f}%)")


def filtrar_conflitos_duas_rodas(deteccoes):
    if len(deteccoes) < 2:
        return deteccoes

    filtradas = []
    ignorar_indices = set()

    for i, (box1, classe1, conf1) in enumerate(deteccoes):
        if i in ignorar_indices:
            continue
        
        for j, (box2, classe2, conf2) in enumerate(deteccoes):
            if i == j or j in ignorar_indices:
                continue
            
            if {classe1, classe2} == {"moto", "bicicleta"} or {classe1, classe2} == {"carro", "caminhao"}:
                x1 = max(box1[0], box2[0])
                y1 = max(box1[1], box2[1])
                x2 = min(box1[2], box2[2])
                y2 = min(box1[3], box2[3])
                
                if x1 < x2 and y1 < y2: 
                    if conf1 >= conf2:
                        ignorar_indices.add(j)
                    else:
                        ignorar_indices.add(i)
                        break
        
        if i not in ignorar_indices:
            filtradas.append((box1, classe1, conf1))
            
    return filtradas

class CapturaRTSP:
    def __init__(self, url):
        self.url = url
        self.quadro = None
        self.rodando = True
        self.cap = None  
        self.bloqueio = threading.Lock()
        self.thread = threading.Thread(target=self._atualizar, daemon=True)
        self.thread.start()

    def _atualizar(self):
        while self.rodando:
            self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            while self.rodando and self.cap.isOpened():
                try:
                    ret, quadro = self.cap.read()
                    if not ret: break
                    with self.bloqueio: self.quadro = quadro
                except Exception: break
            if self.cap is not None: self.cap.release()
            for _ in range(20):
                if not self.rodando: break
                time.sleep(0.1)

    def ler_quadro(self):
        with self.bloqueio: return self.quadro.copy() if self.quadro is not None else None

    def parar(self):
        self.rodando = False
        if self.cap is not None:
            try: self.cap.release()
            except Exception: pass
        self.thread.join(timeout=1.0)  

def desenhar_painel_lateral(altura: int) -> np.ndarray:
    painel = np.full((altura, LARGURA_PAINEL_LATERAL, 3), (18, 18, 18), dtype=np.uint8)
    cv2.putText(painel, "[", (10, 30), cv2.FONT_HERSHEY_PLAIN, 1, (255, 140, 0), 1, cv2.LINE_AA)
    cv2.putText(painel, "]", (LARGURA_PAINEL_LATERAL - 20, 30), cv2.FONT_HERSHEY_PLAIN, 1, (255, 140, 0), 1, cv2.LINE_AA)
    cv2.putText(painel, "C.I.S. OBJETOS - Monitoramento", (25, 30), cv2.FONT_HERSHEY_DUPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(painel, "DETECTOR IA GLOBAL ATIVO", (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 210, 0), 1, cv2.LINE_AA)
    cv2.line(painel, (15, 55), (LARGURA_PAINEL_LATERAL - 15, 55), (40, 40, 40), 1)
    
    y_card = 65
    cv2.rectangle(painel, (15, y_card), (LARGURA_PAINEL_LATERAL - 15, y_card + 165), (28, 28, 28), -1)
    cv2.rectangle(painel, (15, y_card), (LARGURA_PAINEL_LATERAL - 15, y_card + 165), (50, 50, 50), 1)
    cv2.putText(painel, "TOTAL DE DETECCOES POR CLASSE", (25, y_card + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (130, 130, 130), 1, cv2.LINE_AA)
    
    linhas_y = y_card + 42
    for i, (classe, total) in enumerate(estatisticas_classes.items()):
        if i >= 7: break 
        cor_texto = CORES_CLASSES.get(classe, (255, 255, 255))
        cv2.putText(painel, f"{classe.upper()}:", (25, linhas_y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(painel, f"{total}", (180, linhas_y), cv2.FONT_HERSHEY_DUPLEX, 0.42, cor_texto, 1, cv2.LINE_AA)
        linhas_y += 16

    y_hist = y_card + 180
    cv2.putText(painel, "HISTORICO DE DETECCOES RECENTES", (20, y_hist), cv2.FONT_HERSHEY_DUPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.line(painel, (15, y_hist + 8), (LARGURA_PAINEL_LATERAL - 15, y_hist + 8), (40, 40, 40), 1)

    y_log = y_hist + 25
    for ev in reversed(historico_eventos):
        classe_ev = ev["objeto"]
        cor_status = CORES_CLASSES.get(classe_ev.lower(), (255, 255, 255))
        
        cv2.rectangle(painel, (15, y_log), (17, y_log + 25), cor_status, -1)
        cv2.putText(painel, ev["hora"], (25, y_log + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 120), 1, cv2.LINE_AA)
        cv2.putText(painel, f"{classe_ev.upper()}", (95, y_log + 16), cv2.FONT_HERSHEY_DUPLEX, 0.38, (230, 230, 230), 1, cv2.LINE_AA)
        cv2.putText(painel, f"({ev['confianca']*100:.0f}%)", (220, y_log + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.35, cor_status, 1, cv2.LINE_AA)
        y_log += 32
        if y_log > altura - 35: break

    cv2.rectangle(painel, (0, altura - 25), (LARGURA_PAINEL_LATERAL, altura), (10, 10, 10), -1)
    cv2.putText(painel, "STATUS: RODANDO // ESC PARA SAIR", (20, altura - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (80, 80, 80), 1, cv2.LINE_AA)
    return painel

print("[INFO] Carregando modelo YOLOv8...")
modelo = YOLO("yolov8m.pt") 

camera = CapturaRTSP(URL_RTSP)
contador_frames = 0
deteccoes_atuais = []

try:
    while True:
        quadro = camera.ler_quadro()
        if quadro is None:
            time.sleep(0.01)
            continue

        contador_frames += 1
        quadro_exibicao = cv2.resize(quadro, (960, 540))

        if contador_frames % PROCESSAR_A_CADA_N_FRAMES == 0:
            resultados = modelo(quadro, verbose=False)[0]
            deteccoes_brutas = []

            if resultados.boxes is not None:
                for box in resultados.boxes:
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])

                    if cls_id in CLASSES_ALVO and conf >= LIMIAR_CONFIANCA:
                        classe_nome = CLASSES_ALVO[cls_id]
                        deteccoes_brutas.append((box.xyxy[0].tolist(), classe_nome, conf))

            deteccoes_atuais = filtrar_conflitos_duas_rodas(deteccoes_brutas)
            
            for bbox, classe, conf in deteccoes_atuais:
                registrar_deteccao(classe, conf, bbox)

        escala_x = quadro_exibicao.shape[1] / quadro.shape[1]
        escala_y = quadro_exibicao.shape[0] / quadro.shape[0]

        for bbox, classe, conf in deteccoes_atuais:
            x1, y1, x2, y2 = bbox
            dx1, dy1 = int(x1 * escala_x), int(y1 * escala_y)
            dx2, dy2 = int(x2 * escala_x), int(y2 * escala_y)
            
            cor = CORES_CLASSES.get(classe, (255, 255, 255))
            cv2.rectangle(quadro_exibicao, (dx1, dy1), (dx2, dy2), cor, 2)
            cv2.putText(quadro_exibicao, f"{classe.upper()} {conf:.2f}", (dx1, dy1 - 8), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, cor, 2, cv2.LINE_AA)

        cv2.putText(quadro_exibicao, "PORTARIA - CAM 19", (15, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        
        painel_lateral = desenhar_painel_lateral(quadro_exibicao.shape[0])
        tela_final = np.hstack([quadro_exibicao, painel_lateral])
        
        cv2.imshow("Sistema de Detecao de Objetos", tela_final)

        if cv2.waitKey(1) & 0xFF == 27: 
            break
        
finally:
    print("[INFO] Finalizando recursos...")
    camera.parar()
    cv2.destroyAllWindows()