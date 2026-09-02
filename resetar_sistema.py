#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LIMPEZA E RESET DO BANCO DE DADOS
Sistema de Reconhecimento Facial v2

Este script oferece opções seguras de limpeza do banco de dados
com backup automático antes de qualquer operação.
"""

import os
import shutil
import sqlite3
from datetime import datetime

# Configurações
BASE_DIR = "database"
DB_PATH = os.path.join(BASE_DIR, "system.db")
BACKUP_DIR = os.path.join(BASE_DIR, "backups")


def criar_backup():
    """Cria backup do banco antes de qualquer operação"""
    os.makedirs(BACKUP_DIR, exist_ok=True)
    
    if os.path.exists(DB_PATH):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(BACKUP_DIR, f"system_backup_{timestamp}.db")
        shutil.copy2(DB_PATH, backup_path)
        print(f"✓ Backup criado: {backup_path}\n")
        return backup_path
    return None


def limpar_historico():
    """Remove apenas o histórico de presença, mantém pessoas cadastradas"""
    print("=" * 60)
    print("OPÇÃO 1: LIMPAR APENAS HISTÓRICO")
    print("=" * 60)
    print("\nIsto vai:")
    print("  ✓ Manter todas as pessoas cadastradas")
    print("  ✓ Manter embeddings (reconhecimento)")
    print("  ✓ DELETAR histórico de presença")
    print("  ✗ Resetar contador 'Hoje'")
    
    confirmacao = input("\nDeseja continuar? (s/n): ").strip().lower()
    
    if confirmacao != 's':
        print("Cancelado.\n")
        return False
    
    backup = criar_backup()
    
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM historico_presenca")
        conn.commit()
        conn.close()
        
        print("✓ Histórico limpo!")
        print("✓ Pessoas mantidas no banco\n")
        return True
    except Exception as e:
        print(f"✗ Erro ao limpar: {e}")
        print(f"✓ Banco restaurado do backup: {backup}\n")
        return False


def resetar_banco_completo():
    """Deleta todo o banco - começa do zero"""
    print("=" * 60)
    print("OPÇÃO 2: RESETAR BANCO COMPLETO")
    print("=" * 60)
    print("\nIsto vai:")
    print("  ✗ DELETAR todas as pessoas cadastradas")
    print("  ✗ DELETAR todos os embeddings")
    print("  ✗ DELETAR histórico de presença")
    print("  ℹ Necessário rodar --cadastro novamente")
    
    confirmacao = input("\n⚠ Tem certeza? (s/n): ").strip().lower()
    
    if confirmacao != 's':
        print("Cancelado.\n")
        return False
    
    double_check = input("⚠ SEGUNDA CONFIRMAÇÃO - Isto não pode ser desfeito! (s/n): ").strip().lower()
    
    if double_check != 's':
        print("Cancelado.\n")
        return False
    
    backup = criar_backup()
    
    try:
        if os.path.exists(DB_PATH):
            os.remove(DB_PATH)
            print("✓ Banco deletado!")
            print("✓ Diretórios mantidos (faces, comparacoes, logs, etc)")
            print(f"✓ Backup salvo em: {backup}")
            print("\nPróximo passo:")
            print("  python reconhecimento_facial_v2.py --cadastro\n")
            return True
    except Exception as e:
        print(f"✗ Erro ao deletar: {e}")
        print(f"✓ Banco restaurado do backup: {backup}\n")
        return False


def resetar_tudo():
    """Deleta TUDO - banco e diretórios de imagens"""
    print("=" * 60)
    print("OPÇÃO 3: RESETAR TUDO (COMPLETO)")
    print("=" * 60)
    print("\nIsto vai DELETAR:")
    print("  ✗ Banco de dados")
    print("  ✗ Todas as pessoas cadastradas")
    print("  ✗ Todas as imagens de referência (faces/)")
    print("  ✗ Todas as comparações (comparacoes/)")
    print("  ✗ Histórico e registros")
    print("  ✓ Logs mantidos (para debug)")
    
    confirmacao = input("\n⚠ Tem certeza? (s/n): ").strip().lower()
    
    if confirmacao != 's':
        print("Cancelado.\n")
        return False
    
    double_check = input("⚠ SEGUNDA CONFIRMAÇÃO - Isto não pode ser desfeito! (s/n): ").strip().lower()
    
    if double_check != 's':
        print("Cancelado.\n")
        return False
    
    backup = criar_backup()
    
    try:
        # Backup dos diretórios também
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        dirs_to_backup = ["faces", "comparacoes", "detections", "capturas"]
        for dir_name in dirs_to_backup:
            dir_path = os.path.join(BASE_DIR, dir_name)
            if os.path.exists(dir_path):
                backup_path = os.path.join(BACKUP_DIR, f"{dir_name}_backup_{timestamp}")
                shutil.copytree(dir_path, backup_path)
        
        # Deletar
        if os.path.exists(DB_PATH):
            os.remove(DB_PATH)
        
        dirs_to_delete = ["faces", "comparacoes", "detections", "capturas"]
        for dir_name in dirs_to_delete:
            dir_path = os.path.join(BASE_DIR, dir_name)
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)
        
        print("✓ Banco deletado!")
        print("✓ Imagens deletadas!")
        print("✓ Diretórios deletados!")
        print("✓ Logs mantidos (para debug)")
        print(f"✓ Backup completo salvo em: {BACKUP_DIR}")
        print("\nPróximo passo:")
        print("  python reconhecimento_facial_v2.py --cadastro\n")
        return True
    except Exception as e:
        print(f"✗ Erro ao deletar: {e}\n")
        return False


def ver_status():
    """Mostra status do banco de dados"""
    print("=" * 60)
    print("STATUS DO BANCO DE DADOS")
    print("=" * 60)
    
    # Verificar arquivo
    if not os.path.exists(DB_PATH):
        print("\n❌ Banco de dados NÃO existe")
        print("   Próximo passo: python reconhecimento_facial_v2.py --cadastro\n")
        return
    
    print("\n✓ Banco de dados existe")
    
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Contar pessoas
        cursor.execute("SELECT COUNT(*) FROM pessoas")
        num_pessoas = cursor.fetchone()[0]
        
        # Contar histórico
        cursor.execute("SELECT COUNT(*) FROM historico_presenca")
        num_registros = cursor.fetchone()[0]
        
        # Contar pessoas distintas hoje
        cursor.execute("""
            SELECT COUNT(DISTINCT pessoa_id) FROM historico_presenca 
            WHERE DATE(data_hora) = DATE('now')
        """)
        presentes_hoje = cursor.fetchone()[0]
        
        # Listar pessoas
        cursor.execute("SELECT id, nome FROM pessoas ORDER BY id")
        pessoas = cursor.fetchall()
        
        conn.close()
        
        print(f"\n📊 ESTATÍSTICAS:")
        print(f"   Pessoas cadastradas: {num_pessoas}")
        print(f"   Total de registros: {num_registros}")
        print(f"   Presentes hoje: {presentes_hoje}")
        
        if pessoas:
            print(f"\n👥 PESSOAS CADASTRADAS:")
            for pid, nome in pessoas:
                print(f"   {pid} - {nome}")
        else:
            print(f"\n⚠ Nenhuma pessoa cadastrada!")
            print(f"   Próximo passo: python reconhecimento_facial_v2.py --cadastro")
        
        print()
        
    except Exception as e:
        print(f"✗ Erro ao ler banco: {e}\n")


def ver_backups():
    """Lista todos os backups disponíveis"""
    print("=" * 60)
    print("BACKUPS DISPONÍVEIS")
    print("=" * 60)
    
    if not os.path.exists(BACKUP_DIR):
        print("\n❌ Nenhum backup disponível\n")
        return
    
    backups = sorted(os.listdir(BACKUP_DIR), reverse=True)
    
    if not backups:
        print("\n❌ Nenhum backup disponível\n")
        return
    
    print()
    for i, backup in enumerate(backups[:10], 1):  # Mostrar últimos 10
        backup_path = os.path.join(BACKUP_DIR, backup)
        size = os.path.getsize(backup_path) / 1024  # KB
        mtime = datetime.fromtimestamp(os.path.getmtime(backup_path))
        print(f"{i}. {backup}")
        print(f"   Tamanho: {size:.1f} KB | Criado: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
    
    print()


def restaurar_backup():
    """Restaura um backup anterior"""
    print("=" * 60)
    print("RESTAURAR BACKUP")
    print("=" * 60)
    
    if not os.path.exists(BACKUP_DIR):
        print("\n❌ Nenhum backup disponível\n")
        return
    
    backups = sorted(os.listdir(BACKUP_DIR), reverse=True)
    
    if not backups:
        print("\n❌ Nenhum backup disponível\n")
        return
    
    print("\nBackups disponíveis:")
    for i, backup in enumerate(backups[:10], 1):
        print(f"{i}. {backup}")
    
    try:
        escolha = input("\nQual backup restaurar? (número): ").strip()
        indice = int(escolha) - 1
        
        if indice < 0 or indice >= len(backups[:10]):
            print("❌ Opção inválida\n")
            return
        
        backup_selecionado = backups[indice]
        backup_path = os.path.join(BACKUP_DIR, backup_selecionado)
        
        confirmacao = input(f"\nRestaurar {backup_selecionado}? (s/n): ").strip().lower()
        
        if confirmacao != 's':
            print("Cancelado.\n")
            return
        
        # Fazer backup do atual antes de restaurar
        if os.path.exists(DB_PATH):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            atual_backup = os.path.join(BACKUP_DIR, f"system_antes_restaurar_{timestamp}.db")
            shutil.copy2(DB_PATH, atual_backup)
            print(f"✓ Banco atual salvo em: {atual_backup}")
        
        # Restaurar
        shutil.copy2(backup_path, DB_PATH)
        print(f"✓ Banco restaurado: {backup_selecionado}\n")
        
    except ValueError:
        print("❌ Entrada inválida\n")
    except Exception as e:
        print(f"✗ Erro ao restaurar: {e}\n")


def menu_principal():
    """Menu principal interativo"""
    while True:
        print("=" * 60)
        print("LIMPEZA E RESET - BANCO DE DADOS")
        print("=" * 60)
        print("\nOpções:")
        print("  1. Ver status do banco")
        print("  2. Ver backups disponíveis")
        print("  3. Restaurar um backup")
        print("  4. Limpar apenas histórico")
        print("  5. Resetar banco completo")
        print("  6. Resetar tudo (banco + imagens)")
        print("  0. Sair")
        
        opcao = input("\nEscolha uma opção (0-6): ").strip()
        
        if opcao == '1':
            ver_status()
        elif opcao == '2':
            ver_backups()
        elif opcao == '3':
            restaurar_backup()
        elif opcao == '4':
            limpar_historico()
        elif opcao == '5':
            resetar_banco_completo()
        elif opcao == '6':
            resetar_tudo() 
        elif opcao == '0':
            print("Saindo...\n")
            break
        else:
            print("❌ Opção inválida\n")


if __name__ == "__main__":
    try:
        menu_principal()
    except KeyboardInterrupt:
        print("\n\nInterrompido pelo usuário\n")
    except Exception as e:
        print(f"\n❌ Erro: {e}\n")