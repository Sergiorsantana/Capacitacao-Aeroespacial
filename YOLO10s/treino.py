# ==========================================
# YOLOv10s 
# ==========================================
# O YOLOv10s é a versão mais leve da família YOLOv10. Foi feito para rodar rápido,
# com baixo consumo de GPU, mantendo boa precisão para tarefas gerais de detecção.

# ==========================================
# YOLOv10s – Treinamento Completo
# ==========================================
# Script responsável por preparar o dataset, ajustar rótulos,
# gerar o YAML, treinar o YOLOv10s e salvar métricas e predições.
#
# O que este pipeline faz:
# 1) Faz o download do dataset via Roboflow já no formato YOLO.
# 2) Corrige os rótulos (classe 1 → 0) para manter apenas uma classe.
# 3) Cria automaticamente o arquivo YAML usado pelo Ultralytics.
# 4) Inicia o treinamento do YOLOv10s com 140 épocas e diversos ajustes:
#    - LR refinado, momentum, decay, freeze e augmentations.
# 5) Lê o CSV gerado pelo Ultralytics e produz gráficos de métricas
#    como precision, mAP50, losses e recall.
# 6) Compacta todos os resultados em um único arquivo ZIP.
# 7) Realiza uma predição de teste e salva a imagem anotada.
#
# YOLOv10s é um detector rápido e leve, ideal quando se busca um 
# bom equilíbrio entre velocidade e precisão. O formato do dataset 
# e a etapa de correção de rótulos garantem compatibilidade total 
# com o pipeline da Ultralytics.


import os
import shutil
import matplotlib.pyplot as plt
import pandas as pd
from roboflow import Roboflow
from ultralytics import YOLO

# -----------------------------
# 1. CONFIGURAÇÕES INICIAIS
# -----------------------------
API_KEY = "6yKQfUumfFPyQzjUodnU"
WORKSPACE = "college-jcb9y"
PROJECT_NAME = "aircraft-damage-detection-a8z4k"
VERSION = 1
FORMAT = "yolov8"
RUN_NAME = "yolov10s_aircraft_damage_140ep"

# -----------------------------
# 2. DOWNLOAD DO DATASET
# -----------------------------
print(f"\n🚀 Iniciando download do dataset '{PROJECT_NAME}' via Roboflow...")

try:
    rf = Roboflow(api_key=API_KEY)
    project = rf.workspace(WORKSPACE).project(PROJECT_NAME)
    dataset = project.version(VERSION).download(FORMAT)
    dataset_path = dataset.location
    print(f"✅ Dataset baixado em: {dataset_path}")
except Exception as e:
    print(f"❌ Erro ao baixar o dataset: {e}")
    exit(1)

# -----------------------------
# 3. CORREÇÃO DE RÓTULOS (Classe 1 → 0)
# -----------------------------
print("\n🔧 Corrigindo rótulos (classe 1 → 0)...")
count_corrections = 0
label_dirs = [
    os.path.join(dataset_path, 'train/labels'),
    os.path.join(dataset_path, 'valid/labels'),
    os.path.join(dataset_path, 'test/labels')
]

for label_dir in label_dirs:
    if os.path.exists(label_dir):
        for filename in os.listdir(label_dir):
            if filename.endswith(".txt"):
                file_path = os.path.join(label_dir, filename)
                with open(file_path, "r") as f:
                    lines = f.readlines()

                new_lines = []
                needs_rewrite = False
                for line in lines:
                    if line.strip().startswith("1 "):
                        new_lines.append("0" + line.strip()[1:] + "\n")
                        count_corrections += 1
                        needs_rewrite = True
                    else:
                        new_lines.append(line)

                if needs_rewrite:
                    with open(file_path, "w") as f:
                        f.writelines(new_lines)

print(f"✅ Correção concluída — {count_corrections} ocorrências alteradas.")

# -----------------------------
# 4. CRIAÇÃO DO ARQUIVO YAML
# -----------------------------
dataset_yaml = f"""
path: {dataset_path}
train: train/images
val: valid/images
test: test/images
nc: 1
names: ["damage"]
"""
yaml_file = os.path.join(dataset_path, "aircraft_dataset.yaml")
with open(yaml_file, "w") as f:
    f.write(dataset_yaml)
print(f"✅ YAML criado em: {yaml_file}")

# -----------------------------
# 5. TREINAMENTO COM YOLOv8s
# -----------------------------
print(f"\n🏋️ Iniciando treinamento YOLOv10s por 140 épocas...\n")

model = YOLO("yolov10s.pt")

model.train(
    data=yaml_file,
    epochs=140,
    batch=16,
    imgsz=768,
    freeze=1,
    name=RUN_NAME,
    lr0=0.007,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.003,
    patience=20,
    flipud=0.4,
    fliplr=0.5,
    hsv_h=0.015,
    hsv_s=0.5,
    hsv_v=0.4
)

# -----------------------------
# 6. GERAÇÃO E SALVAMENTO DE GRÁFICOS
# -----------------------------
print("\n📊 Gerando gráficos de métricas...")

runs_dir = os.path.join("runs", "detect", RUN_NAME)
csv_path = os.path.join(runs_dir, "results.csv")

if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    # Precision e mAP50
    plt.figure(figsize=(10, 6))
    plt.plot(df["epoch"], df["metrics/precision(B)"], label="Precision (Val)", color="blue")
    plt.plot(df["epoch"], df["metrics/mAP50(B)"], label="mAP50 (Val)", color="red")
    plt.xlabel("Épocas")
    plt.ylabel("Métrica de Validação")
    plt.title("Evolução de Precision e mAP50")
    plt.legend()
    plt.grid(True)

    graph_path = os.path.join(runs_dir, "precision_map50.png")
    plt.savefig(graph_path)
    plt.close()
    print(f"✅ Gráfico Precision/mAP50 salvo em: {graph_path}")

    # Losses
    plt.figure(figsize=(10, 6))
    plt.plot(df["epoch"], df["train/box_loss"], label="Box Loss (Train)")
    plt.plot(df["epoch"], df["train/cls_loss"], label="Class Loss (Train)")
    plt.plot(df["epoch"], df["metrics/recall(B)"], label="Recall (Val)")
    plt.xlabel("Épocas")
    plt.ylabel("Valores")
    plt.title("Evolução de Losses e Recall")
    plt.legend()
    plt.grid(True)

    loss_graph_path = os.path.join(runs_dir, "loss_recall.png")
    plt.savefig(loss_graph_path)
    plt.close()
    print(f"✅ Gráfico Loss/Recall salvo em: {loss_graph_path}")

else:
    print("⚠️ CSV de resultados não encontrado. Nenhum gráfico foi gerado.")

# -----------------------------
# 7. EXPORTAÇÃO FINAL
# -----------------------------
zip_name = f"{RUN_NAME}_results"
print("\n📦 Compactando resultados...")
shutil.make_archive(zip_name, 'zip', os.path.join("runs", "detect"), RUN_NAME)
print(f"✅ Arquivo ZIP criado: {zip_name}.zip")

print("\n🎯 Execução finalizada com sucesso.")

# -----------------------------
# 6. PREDIÇÃO DE TESTE
# -----------------------------
print("\n🔍 Realizando predição de teste...")

test_image = os.path.join(dataset_path, "test/images", os.listdir(os.path.join(dataset_path, "test/images"))[0])

# Pasta de salvamento da predição
save_dir = os.path.join("runs", "detect", RUN_NAME, "test_prediction")
os.makedirs(save_dir, exist_ok=True)

# Predição e salvamento
results = model.predict(
    source=test_image,
    save=True,           # salva imagem com bounding boxes
    save_txt=False,
    project=os.path.join("runs", "detect", RUN_NAME),
    name="test_prediction"
)

print(f"✅ Predição de teste salva em: {save_dir}")
