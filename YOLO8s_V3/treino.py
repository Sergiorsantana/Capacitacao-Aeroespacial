# ==========================================
# SCRIPT CONSOLIDADO - YOLOv8s (2 ETAPAS) HYBRID BEAST MODE
# ==========================================

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
RUN_NAME = "yolov8s_aircraft_damage_100ep_hybrid"

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
# 5. TREINAMENTO - ETAPA 1 (Freeze=1)
# -----------------------------
print(f"\n🏋️ Etapa 1 — Treinando YOLOv8s (congelando backbone, freeze=1)...\n")

model = YOLO("yolov8s.pt")

model.train(
    data=yaml_file,
    epochs=80,
    batch=16,
    imgsz=640,
    freeze=1,
    name=RUN_NAME,
    optimizer="AdamW",
    lr0=0.001,
    lrf=0.01,
    weight_decay=0.001,
    patience=10,
    augment=True,
    flipud=0.5,
    fliplr=0.7,
    hsv_h=0.03,
    hsv_s=0.7,
    hsv_v=0.6,
    scale=0.5,
    translate=0.5,
    shear=0.5,
    mosaic=1.0,
    mixup=0.5,
    copy_paste=0.5
)

# -----------------------------
# 6. TREINAMENTO - ETAPA 2 (Freeze=0)
# -----------------------------
print(f"\n🏋️ Etapa 2 — Descongelando tudo (freeze=0, fine-tuning completo)...\n")

best_weights = os.path.join("runs", "detect", RUN_NAME, "weights", "best.pt")
model = YOLO(best_weights)

model.train(
    data=yaml_file,
    epochs=120,
    batch=16,
    imgsz=640,
    freeze=0,
    name=RUN_NAME + "_stage2",
    optimizer="AdamW",
    lr0=0.0005,
    lrf=0.005,
    weight_decay=0.001,
    patience=10,
    augment=True,
    flipud=0.5,
    fliplr=0.7,
    hsv_h=0.03,
    hsv_s=0.7,
    hsv_v=0.6,
    scale=0.5,
    translate=0.5,
    shear=0.5,
    mosaic=1.0,
    mixup=0.5,
    copy_paste=0.5
)

# -----------------------------
# 7. PREDIÇÃO DE TESTE
# -----------------------------
print("\n🔍 Realizando predição de teste...")

test_image_dir = os.path.join(dataset_path, "test/images")
test_image_name = os.listdir(test_image_dir)[0]
test_image = os.path.join(test_image_dir, test_image_name)

pred_dir = os.path.join("runs", "detect", RUN_NAME + "_stage2", "test_pred")
os.makedirs(pred_dir, exist_ok=True)

results = model.predict(
    source=test_image,
    save=True,
    save_txt=False,
    project=pred_dir,
    name="prediction",
    exist_ok=True
)
print(f"✅ Predição de teste salva em: {pred_dir}")

# -----------------------------
# 8. GERAÇÃO E SALVAMENTO DE GRÁFICOS
# -----------------------------
print("\n📊 Gerando gráficos de métricas...")

runs_dir = os.path.join("runs", "detect", RUN_NAME + "_stage2")
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
    plt.title("Evolução de Precision e mAP50 (Etapa 2)")
    plt.legend()
    plt.grid(True)
    graph_path = os.path.join(runs_dir, "precision_map50.png")
    plt.savefig(graph_path)
    plt.close()
    print(f"✅ Gráfico Precision/mAP50 salvo em: {graph_path}")

    # Losses e Recall
    plt.figure(figsize=(10, 6))
    plt.plot(df["epoch"], df["train/box_loss"], label="Box Loss (Train)")
    plt.plot(df["epoch"], df["train/cls_loss"], label="Class Loss (Train)")
    plt.plot(df["epoch"], df["metrics/recall(B)"], label="Recall (Val)")
    plt.xlabel("Épocas")
    plt.ylabel("Valores")
    plt.title("Evolução de Losses e Recall (Etapa 2)")
    plt.legend()
    plt.grid(True)
    loss_graph_path = os.path.join(runs_dir, "loss_recall.png")
    plt.savefig(loss_graph_path)
    plt.close()
    print(f"✅ Gráfico Loss/Recall salvo em: {loss_graph_path}")
else:
    print("⚠️ CSV de resultados não encontrado. Nenhum gráfico foi gerado.")

# -----------------------------
# 9. EXPORTAÇÃO FINAL
# -----------------------------
zip_name = f"{RUN_NAME}_final_results"
print("\n📦 Compactando resultados...")
shutil.make_archive(zip_name, 'zip', os.path.join("runs", "detect"))
print(f"✅ Arquivo ZIP criado: {zip_name}.zip")

print("\n🎯 Execução finalizada com sucesso — YOLOv8s HYBRID BEAST MODE completo!")
