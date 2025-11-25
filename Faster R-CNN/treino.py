# ==========================================
# SCRIPT CONSOLIDADO - RT-DETR-X (GPU 6–8 GB)
# ==========================================

import os
import shutil
import matplotlib.pyplot as plt
import pandas as pd
from roboflow import Roboflow
from ultralytics import RTDETR
import torch

# 🔧 Otimização para evitar CUDA OOM
torch.cuda.empty_cache()
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# -----------------------------
# 1. CONFIGURAÇÕES INICIAIS
# -----------------------------
API_KEY = "6yKQfUumfFPyQzjUodnU"
WORKSPACE = "college-jcb9y"
PROJECT_NAME = "aircraft-damage-detection-a8z4k"
VERSION = 1
FORMAT = "yolov8"
RUN_NAME = "rtdetr_X_aircraft_damage_140ep"  # ✅ Atualizado para X

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
# 3. CORREÇÃO DE RÓTULOS (classe 1 → 0)
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
# 4. ARQUIVO YAML
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
# 5. TREINAMENTO — RT-DETR-X
# -----------------------------
print("\n⬇️ Carregando RT-DETR-X pretrained (Ultralytics)...")
# ✅ Modelo mais leve - ideal para 6-8GB VRAM
model = RTDETR("rtdetr-x.pt")

print(f"\n🏋️ Iniciando treinamento RT-DETR (X) por 140 épocas...\n")

model.train(
    data=yaml_file,
    epochs=150,
    batch=2,
    imgsz=640,
    name=RUN_NAME,
    device=0,

    lr0=0.00005,
    lrf=0.01,

    patience=50,

    mosaic=0.1,
    mixup=0.0,
    flipud=0.0,
    fliplr=0.3,

    hsv_h=0.005,
    hsv_s=0.2,
    hsv_v=0.2,

    amp=False,
    workers=2,
    cache=False
)



# -----------------------------
# 6. GRÁFICOS
# -----------------------------
print("\n📊 Gerando gráficos de métricas...")

runs_dir = os.path.join("runs", "detect", RUN_NAME)
csv_path = os.path.join(runs_dir, "results.csv")

if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    print("📌 Colunas encontradas no CSV:", list(df.columns))

    # ===========================
    # Ajuste de nomes possíveis
    # ===========================
    col_precision = "precision(B)" if "precision(B)" in df.columns else "metrics/precision"
    col_map50 = "metrics/mAP50(B)" if "metrics/mAP50(B)" in df.columns else "metrics/mAP50"
    col_recall = "recall(B)" if "recall(B)" in df.columns else "metrics/recall"

    col_box_loss = "train/box_loss" if "train/box_loss" in df.columns else "train/loss"
    col_cls_loss = "train/cls_loss" if "train/cls_loss" in df.columns else None

    # ---------------------------
    # GRÁFICO 1 — Precision & mAP50
    # ---------------------------
    plt.figure(figsize=(10, 6))

    if col_precision in df:
        plt.plot(df["epoch"], df[col_precision], label="Precision (Val)")

    if col_map50 in df:
        plt.plot(df["epoch"], df[col_map50], label="mAP50 (Val)")

    plt.xlabel("Épocas")
    plt.ylabel("Métricas de Validação")
    plt.title("Precision e mAP50 (RT-DETR-X)")
    plt.legend()
    plt.grid(True)

    graph_path = os.path.join(runs_dir, "precision_map50.png")
    plt.savefig(graph_path)
    plt.close()
    print(f"✅ Gráfico salvo: {graph_path}")

    # ---------------------------
    # GRÁFICO 2 — Loss & Recall
    # ---------------------------
    plt.figure(figsize=(10, 6))

    if col_box_loss in df:
        plt.plot(df["epoch"], df[col_box_loss], label="Box Loss")

    if col_cls_loss and col_cls_loss in df:
        plt.plot(df["epoch"], df[col_cls_loss], label="Class Loss")

    if col_recall in df:
        plt.plot(df["epoch"], df[col_recall], label="Recall (Val)")

    plt.xlabel("Épocas")
    plt.ylabel("Valores")
    plt.title("Loss e Recall (RT-DETR-X)")
    plt.legend()
    plt.grid(True)

    loss_graph_path = os.path.join(runs_dir, "loss_recall.png")
    plt.savefig(loss_graph_path)
    plt.close()
    print(f"✅ Gráfico salvo: {loss_graph_path}")

else:
    print("⚠️ CSV não encontrado.")

# -----------------------------
# 7. EXPORTAÇÃO
# -----------------------------
zip_name = f"{RUN_NAME}_results"
print("\n📦 Compactando resultados...")
shutil.make_archive(zip_name, 'zip', os.path.join("runs", "detect"), RUN_NAME)
print(f"✅ ZIP criado: {zip_name}.zip")

# -----------------------------
# 8. PREDIÇÃO FINAL
# -----------------------------
print("\n🔍 Realizando predição de teste...")

test_dir = os.path.join(dataset_path, "test/images")
test_image = os.path.join(test_dir, os.listdir(test_dir)[0])

save_dir = os.path.join("runs", "detect", RUN_NAME, "test_prediction")
os.makedirs(save_dir, exist_ok=True)

model.predict(
    source=test_image,
    save=True,
    project=os.path.join("runs", "detect", RUN_NAME),
    name="test_prediction"
)

print(f"✅ Predição salva em: {save_dir}")

# -----------------------------
# 9. RESUMO FINAL
# -----------------------------
print("\n" + "="*50)
print("🎯 TREINAMENTO CONCLUÍDO - RT-DETR-X")
print("="*50)
print(f"📁 Resultados: runs/detect/{RUN_NAME}")
print(f"📦 ZIP: {zip_name}.zip")
print(f"🔍 Predição teste: {save_dir}")
print("="*50)
