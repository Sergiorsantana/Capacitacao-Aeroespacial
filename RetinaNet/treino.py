# ==========================================
# RETINANET 
# ==========================================
# O RetinaNet é um modelo de detecção de objetos baseado em uma única rede.
# Ele usa a Loss Focal para lidar melhor com o desequilíbrio entre classes,
# o que ajuda a melhorar a precisão em objetos menores ou menos frequentes.

# ==========================================
# RETINANET 
# ==========================================
# Pipeline 
#
# Resumo do que o script faz:
# 1) Baixa o dataset via Roboflow no formato COCO (requerido pelo Detectron2).
# 2) Converte as anotações para o formato interno usado pelo framework.
# 3) Registra automaticamente os splits (train/valid/test) no DatasetCatalog.
# 4) Carrega o RetinaNet R50-FPN pré-treinado no COCO e ajusta para 1 classe.
# 5) Configura hiperparâmetros (LR, warmup, steps, épocas, batch size etc.).
# 6) Inicia o treinamento e salva checkpoints no diretório do experimento.
# 7) Avalia o modelo usando o COCOEvaluator no conjunto de validação.
# 8) Gera visualizações das predições nas imagens de teste.
# 9) Salva métricas em JSON e compacta todo o resultado em um arquivo ZIP.
#
# Observações gerais:
# - O RetinaNet é um detector de uma só passada (one-stage).
# - Ele usa a Focal Loss, que melhora o aprendizado em datasets desbalanceados,
#   evitando que a rede foque demais em exemplos "fáceis".
# - Detectron2 trabalha naturalmente com COCO, por isso a conversão inicial.
# - O pipeline gera métricas, imagens anotadas e um ZIP pronto para análise.


import os
import shutil
import json
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from roboflow import Roboflow
import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import Visualizer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

# 🔧 Evitar OOM
torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# -----------------------------
# CONFIGS
# -----------------------------
API_KEY = "6yKQfUumfFPyQzjUodnU"
WORKSPACE = "college-jcb9y"
PROJECT_NAME = "aircraft-damage-detection-a8z4k"
VERSION = 1
FORMAT = "coco"  # Detectron2 usa formato COCO
RUN_NAME = "retinanet_aircraft_damage_140ep"
OUTPUT_DIR = f"./output/{RUN_NAME}"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# BAIXAR DATASET
# -----------------------------
print(f"\n🚀 Baixando dataset {PROJECT_NAME} no formato COCO...")

rf = Roboflow(api_key=API_KEY)
project = rf.workspace(WORKSPACE).project(PROJECT_NAME)
dataset = project.version(VERSION).download(FORMAT)
dataset_path = dataset.location

print(f"✅ Dataset baixado em: {dataset_path}")

# -----------------------------
# CONVERTER DATASET PARA DETECTRON2
# -----------------------------
def get_aircraft_dicts(img_dir, ann_file):
    """Converte anotações COCO para formato Detectron2"""
    with open(ann_file) as f:
        coco_data = json.load(f)
    
    # Criar mapeamento image_id -> filename
    imgs = {img['id']: img for img in coco_data['images']}
    
    dataset_dicts = []
    for ann in coco_data['annotations']:
        if ann['image_id'] not in imgs:
            continue
            
        img_info = imgs[ann['image_id']]
        record = {}
        
        filename = os.path.join(img_dir, img_info['file_name'])
        if not os.path.exists(filename):
            continue
            
        record["file_name"] = filename
        record["image_id"] = ann['image_id']
        record["height"] = img_info['height']
        record["width"] = img_info['width']
        
        # Converter bbox de COCO [x,y,w,h] para [x1,y1,x2,y2]
        x, y, w, h = ann['bbox']
        obj = {
            "bbox": [x, y, x + w, y + h],
            "bbox_mode": BoxMode.XYXY_ABS,
            "category_id": 0,  # Classe única: damage
        }
        
        if "objs" not in record:
            record["annotations"] = []
        record["annotations"].append(obj)
        
        # Evitar duplicatas
        if not any(d.get("image_id") == record["image_id"] for d in dataset_dicts):
            dataset_dicts.append(record)
    
    return dataset_dicts

# Registrar datasets
for split in ["train", "valid", "test"]:
    dataset_name = f"aircraft_{split}"
    img_dir = os.path.join(dataset_path, split)
    ann_file = os.path.join(dataset_path, split, "_annotations.coco.json")
    
    if os.path.exists(ann_file):
        DatasetCatalog.register(
            dataset_name, 
            lambda d=img_dir, a=ann_file: get_aircraft_dicts(d, a)
        )
        MetadataCatalog.get(dataset_name).set(thing_classes=["damage"])
        print(f"✅ Registrado: {dataset_name}")

# -----------------------------
# CONFIGURAR RETINANET
# -----------------------------
print("\n⬇️ Configurando RetinaNet pretrained...")

cfg = get_cfg()
# RetinaNet R50-FPN pré-treinado no COCO
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/retinanet_R_50_FPN_3x.yaml")

# Datasets
cfg.DATASETS.TRAIN = ("aircraft_train",)
cfg.DATASETS.TEST = ("aircraft_valid",)

# Hiperparâmetros
cfg.DATALOADER.NUM_WORKERS = 4
cfg.SOLVER.IMS_PER_BATCH = 8
cfg.SOLVER.BASE_LR = 0.001
cfg.SOLVER.MAX_ITER = 140 * 100  # 140 épocas * ~100 batches (ajuste conforme dataset)
cfg.SOLVER.STEPS = (8000, 12000)
cfg.SOLVER.GAMMA = 0.1
cfg.SOLVER.WARMUP_ITERS = 500

# Modelo
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Apenas "damage"
cfg.MODEL.RETINANET.NUM_CLASSES = 1

# Output
cfg.OUTPUT_DIR = OUTPUT_DIR
cfg.SOLVER.CHECKPOINT_PERIOD = 1000

# Avaliação
cfg.TEST.EVAL_PERIOD = 500

print("✅ Configuração RetinaNet pronta!")

# -----------------------------
# TREINAMENTO
# -----------------------------
print("\n🏋️ Treinando RetinaNet...\n")

trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

print("\n✅ Treinamento concluído!")

# -----------------------------
# AVALIAÇÃO
# -----------------------------
print("\n📊 Avaliando modelo no conjunto de validação...")

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

from detectron2.engine import DefaultPredictor
predictor = DefaultPredictor(cfg)

evaluator = COCOEvaluator("aircraft_valid", cfg, False, output_dir=OUTPUT_DIR)
val_loader = build_detection_test_loader(cfg, "aircraft_valid")
results = inference_on_dataset(predictor.model, val_loader, evaluator)

print(f"\n📈 Resultados de Avaliação:")
print(json.dumps(results, indent=2))

# Salvar métricas
with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
    json.dump(results, f, indent=2)

# -----------------------------
# VISUALIZAÇÃO
# -----------------------------
print("\n🖼️ Gerando visualizações de predições...")

test_dataset = get_aircraft_dicts(
    os.path.join(dataset_path, "test"),
    os.path.join(dataset_path, "test", "_annotations.coco.json")
)

aircraft_metadata = MetadataCatalog.get("aircraft_test")

for i, d in enumerate(test_dataset[:5]):  # Visualizar 5 imagens
    img = cv2.imread(d["file_name"])
    outputs = predictor(img)
    
    v = Visualizer(
        img[:, :, ::-1],
        metadata=aircraft_metadata,
        scale=1.0
    )
    
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    result_img = out.get_image()[:, :, ::-1]
    
    output_path = os.path.join(OUTPUT_DIR, f"prediction_{i}.jpg")
    cv2.imwrite(output_path, result_img)
    print(f"✅ Salva: {output_path}")

# -----------------------------
# ZIP FINAL
# -----------------------------
print("\n📦 Criando arquivo ZIP dos resultados...")

zip_name = f"{RUN_NAME}_results"
shutil.make_archive(zip_name, 'zip', OUTPUT_DIR)
print(f"✅ ZIP criado: {zip_name}.zip")

print("\n🎉 Pipeline RetinaNet concluído!")
print(f"📁 Resultados em: {OUTPUT_DIR}")
print(f"📊 Métricas em: {os.path.join(OUTPUT_DIR, 'metrics.json')}")
