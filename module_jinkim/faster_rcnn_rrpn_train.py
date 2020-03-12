# -*- coding: utf-8 -*-

from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.evaluation import inference_on_dataset, RotatedCOCOEvaluator, DatasetEvaluators
from detectron2 import model_zoo
from detectron2.data import transforms as T
from detectron2.data import detection_utils as utils
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader, build_detection_train_loader
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
import os, torch, copy, random, cv2, math, pdb
import numpy as np
import pandas as pd
from tqdm import tqdm
from detectron2.structures import (
    BitMasks,
    Boxes,
    BoxMode,
    Instances,
    Keypoints,
    PolygonMasks,
    RotatedBoxes,
    polygons_to_bitmask,
)
from detectron2.config import CfgNode as CN
from module_jinkim import ADDDatasetMapper


def backward_convert(coordinate, with_label=False):
    """
    :param coordinate: format [x1, y1, x2, y2, x3, y3, x4, y4, (label)] # type ndarray
    :param with_label: default True
    :return: format [x_c, y_c, w, h, theta, (label)]
    """
    boxes = []
    if with_label:
        for rect in coordinate:
            box = np.int0(rect[:-1])
            box = box.reshape([4, 2])
            rect1 = cv2.minAreaRect(box)

            x, y, w, h, theta = rect1[0][0], rect1[0][1], rect1[1][0], rect1[1][1], rect1[2]
            boxes.append([x, y, w, h, theta, rect[-1]])

    else:
        for rect in coordinate:
            box = np.int0(rect)
            box = box.reshape([4, 2])
            rect1 = cv2.minAreaRect(box)

            x, y, w, h, theta = rect1[0][0], rect1[0][1], rect1[1][0], rect1[1][1], rect1[2]
            boxes.append([x, y, w, h, theta])

    return np.array(boxes, dtype=np.float32)


def seperate_train_val(train_val_ratio, origin_ADD_csv, train_csv_path, val_csv_path):
    # 원본 ADD csv 파일을 trining set annotation csv와 validation set annotation csv로 나눠서 저장한다.
    df = pd.read_csv(origin_ADD_csv)
    img_names = (df.groupby('file_name').count()).reset_index()['file_name']
    img_num = img_names.shape[0]

    val_df = pd.DataFrame()
    train_df = pd.DataFrame()
    for i in range(img_num):
        img_name = img_names.iloc[i]
        data_per_img = df[df['file_name'] == img_name]
        k = random.uniform(0, 1)  # 0에서 1사이
        if k < train_val_ratio:
            val_df = val_df.append(data_per_img, ignore_index=True)
        else:
            train_df = train_df.append(data_per_img, ignore_index=True)
    val_df.to_csv(val_csv_path, mode='w', index=False)
    train_df.to_csv(train_csv_path, mode='w', index=False)


def get_ADDxywht_dicts(dataset_dir, val_or_train):
    csv_file = os.path.join(dataset_dir, "ADD_" + val_or_train + ".csv")
    img_dir = os.path.join(dataset_dir, 'images')
    df = pd.read_csv(csv_file)
    image_shape = input_image_scale

    # XYWHA 데이터 생성하기
    xywha = backward_convert(df.loc[:, 'point1_x':'point4_y'].values)
    sub_df = pd.DataFrame(data=xywha, columns=np.array(['center_x', 'center_y', 'width', 'height', 'angle']))
    df = df.join(sub_df)
    df['angle'] = -1 * df['angle']  # cv2의 minarearect랑 표현 방식이 다르다. detectron2는 ccw라고 함. 근데 -1을 곱해야 맞음
    # # ADD label의 좌표가 음수거나 3000보다 클 경우 드랍.
    # cond1 = (0 < df['point1_x']) & (df['point1_x'] < 3000) & (0 < df['point1_y']) & (df['point1_y'] < 3000)
    # cond2 = (0 < df['point2_x']) & (df['point2_x'] < 3000) & (0 < df['point2_y']) & (df['point2_y'] < 3000)
    # cond3 = (0 < df['point3_x']) & (df['point3_x'] < 3000) & (0 < df['point3_y']) & (df['point3_y'] < 3000)
    # cond4 = (0 < df['point4_x']) & (df['point4_x'] < 3000) & (0 < df['point4_y']) & (df['point4_y'] < 3000)
    #
    # df = df[cond1 & cond2 & cond3 & cond4]

    df = (df.reset_index()).drop(columns='index')  # annotation이 몇개 빠지기 때문에 reset index 해준다.

    file_name_dict = {}
    for i in tqdm(range(df.shape[0]),desc='get ADD '+val_or_train + " dataset upload"):
        filename = os.path.join(img_dir, df.loc[i, 'file_name'])
        # height, width = cv2.imread(filename).shape[:2]
        height = image_shape
        width = image_shape
        if filename not in file_name_dict:
            file_name_dict[filename] = dict.fromkeys(["image_id", "height", "width", "annotations"])
            image_id = len(file_name_dict)
            file_name_dict[filename]["image_id"] = image_id
            file_name_dict[filename]["height"] = height
            file_name_dict[filename]["width"] = width
            file_name_dict[filename]["annotations"] = []

        obj = {
            "bbox": df.loc[i, 'center_x':'angle'].values.tolist(),
            "bbox_mode": BoxMode.XYWHA_ABS,
            "category_id": int(df.loc[i, 'class_id']) - 1,
            # detectron2 에서는 class number값이 배경을 나타내게 되어있다. 원래 클래스 아이디에서 하나 뺌
            "iscrowd": 0
        }
        # if math.floor(10 * (i - 1) / df.shape[0]) != math.floor(10 * i / df.shape[0]):
        #     print(i / df.shape[0])

        file_name_dict[filename]["annotations"].append(obj)

    dataset_dicts = []
    for filename in file_name_dict.keys():
        record = file_name_dict[filename]
        record['file_name'] = filename
        dataset_dicts.append(record)
    return dataset_dicts


########### dataset setting part ###########
from detectron2.data import DatasetCatalog, MetadataCatalog

dataset_dir = os.path.expanduser('~/ADD_dataset/train/')  # symbolik link를 유저 path 바로 밑에 설치.(ex :sudo ln -sT ~/hddu/dataset_ADD_20191122/ ~/ADD_dataset)
origin_ADD_csv = os.path.join(dataset_dir, 'labels.csv')
train_csv_path = os.path.join(dataset_dir, 'ADD_train.csv')
val_csv_path = os.path.join(dataset_dir, 'ADD_val.csv')

## divide ADD label csv to train & validation and save label
seperate_train_val(0.18, origin_ADD_csv, train_csv_path, val_csv_path)

for d in ["train", "val"]:
    DatasetCatalog.register("ADDxywht_" + d, lambda d=d: get_ADDxywht_dicts(dataset_dir, d))
    MetadataCatalog.get("ADDxywht_" + d).set(
        thing_classes=['container', 'oil tanker', 'aircraft carrier', 'maritime vessels'])

##################################################################

########### dataset의 gt가 올바르게 등록되었는지 확인하는 코드 ###########
# dataset_dicts = get_ADDxywht_dicts(dataset_dir,'train')
# for d in random.sample(dataset_dicts, 5):
#     img = cv2.imread(d["file_name"])
#     visualizer = Visualizer(img[:, :, ::-1], metadata=ADDxywht_metadata, scale=0.5)
#     labels = [x["category_id"] for x in d['annotations']]
#     boxes = [x["bbox"] for x in d['annotations']]
#     names = visualizer.metadata.get("thing_classes", None)
#     if names:
#         labels = [names[i] for i in labels]
#     labels = [
#         "{}".format(i) + ("|crowd" if a.get("iscrowd", 0) else "")
#         for i, a in zip(labels, d['annotations'])
#     ]
#     visualizer.overlay_instances(labels=labels, boxes=boxes)
#     vis = visualizer.output
#     cv2.imshow('test',vis.get_image()[:, :, ::-1])
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
##################################################################

############################## config file setting #################################################
ClassCount = 4
input_image_scale = 3000 # ADD dataset crop 안했을 때 image scale
max_image_resize = 800  # 이 코드에서 maximum 인풋이미지의 사이즈는 이것으로 결정된다.
num_of_training_imgs = 1300 # ADD dataset crop 안했을 때 training image개수
imgs_per_batch = 1 # 이 코드에서 쓰일 batch size
iter_per_epoch = int(num_of_training_imgs/imgs_per_batch)
iter_alpha = 0 # 추가적으로 iteration 돌리고 싶을 때 이 값을 추가한다.(전체 iteration과 lr이 감소하는 타이밍이 linear하게 늘어남.)
resume_training = False # Decide whether to continue training.
epoch_per_savemodel = 1 # saving per epoch

cfg = get_cfg()
cfg.OUTPUT_DIR = './module_jinkim/output'
configfile = "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"
cfg.merge_from_file(model_zoo.get_config_file(configfile))

if resume_training: # Resume
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # Resume
else:
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(configfile)  # Let training initialize from model zoo

cfg.DATASETS.TRAIN = (['ADDxywht_train'])
cfg.DATASETS.TEST = (['ADDxywht_val'])
# Maximum size of the side of the image during training
cfg.INPUT.MAX_SIZE_TRAIN = max_image_resize
# Size of the smallest side of the image during training
cfg.INPUT.MIN_SIZE_TRAIN = (
max_image_resize - 400, max_image_resize - 300, max_image_resize - 200, max_image_resize - 100, max_image_resize)
# Size of the smallest side of the image during testing. Set to zero to disable resize in testing.
cfg.INPUT.MIN_SIZE_TEST = max_image_resize
# Maximum size of the side of the image during testing
cfg.INPUT.MAX_SIZE_TEST = max_image_resize

cfg.DATALOADER.NUM_WORKERS = 4

cfg.TEST.EVAL_PERIOD = iter_per_epoch # evaluation code testing 하고 싶으면 이 숫자를 조절하면 된다.
# scheduler & learning rate (기존의 것에 비례 혹은 반비례 하게 설정함)
cfg.SOLVER.CHECKPOINT_PERIOD = iter_per_epoch*epoch_per_savemodel
batch_rate = imgs_per_batch/cfg.SOLVER.IMS_PER_BATCH
cfg.SOLVER.STEPS = (int(cfg.SOLVER.STEPS[0]/batch_rate)+iter_alpha, int(cfg.SOLVER.STEPS[1]/batch_rate)+iter_alpha) # 기존의 scheduler setting을 따르되, 배치 사이즈에 따라 늘리고 줄인다.
cfg.SOLVER.MAX_ITER = int(cfg.SOLVER.MAX_ITER/batch_rate)+iter_alpha # 기존의 scheduler setting을 따르되, 배치 사이즈에 반비례 하여 늘리고 줄인다.
cfg.SOLVER.BASE_LR = cfg.SOLVER.BASE_LR*batch_rate # default setting과 batch size를 비교하여 lr을 자동 설정.
cfg.SOLVER.IMS_PER_BATCH = imgs_per_batch # 얘는 맨 나중에 설정! batch_rate 설정과 순서 뒤바뀌지 말것.


cfg.MODEL.MASK_ON = False
cfg.MODEL.ROI_HEADS.NAME = "RROIHeads"
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512  # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = ClassCount
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model

cfg.MODEL.PROPOSAL_GENERATOR.NAME = "RRPN"

cfg.MODEL.RPN.HEAD_NAME = "StandardRPNHead"
cfg.MODEL.RPN.BBOX_REG_WEIGHTS = (1, 1, 1, 1, 1)
# R3det anchor setting
anrsc_p_pmd = [1,2**(1/3),2**(2/3)] # anchor scales per pyramid
cfg.MODEL.ANCHOR_GENERATOR.NAME = "RotatedAnchorGenerator"
cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[32*x for x in anrsc_p_pmd], [64*x for x in anrsc_p_pmd], [128*x for x in anrsc_p_pmd], [256*x for x in anrsc_p_pmd], [512*x for x in anrsc_p_pmd]]
cfg.MODEL.ANCHOR_GENERATOR.ANGLES = [[15, 30, 45, 60,75, 90]]
cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.2, 1/3, 0.5, 2.0, 3, 5]]  # 수정

cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE = "ROIAlignRotated"
cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS = (10.0, 10.0, 5.0, 5.0, 10.0)

# cfg.TEST.AUG = CN({"ENABLED": False})
# cfg.TEST.AUG.MIN_SIZES = (400, 500, 600, 700, 800, 900, 1000, 1100, 1200)
# cfg.TEST.AUG.MAX_SIZE = 4000
# cfg.TEST.AUG.FLIP = True
#
# cfg.TEST.PRECISE_BN = CN({"ENABLED": False})
# cfg.TEST.PRECISE_BN.NUM_ITER = 200


os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
print('FINAL CONFIG FILE')
print(cfg)
########################################################################################################

########### training part ###########

# 이거를 DefaultTrainer를 상속한 다른 클래스로 고치자. 예시 : https://github.com/facebookresearch/detectron2/blob/master/projects/DensePose/train_net.py
class MyTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluators = [RotatedCOCOEvaluator(dataset_name, cfg, True, output_folder)]
        return DatasetEvaluators(evaluators)

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=ADDDatasetMapper(cfg,True))

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name, mapper=ADDDatasetMapper(cfg, False))


trainer = MyTrainer(cfg)
trainer.resume_or_load(resume=resume_training)
trainer.train()
##################################################################

########### evaluation 파트 ###########
# trainer = MyTrainer(cfg)
# trainer.resume_or_load(resume=True)
# trainer.test(cfg, trainer.model)
##################################################################

########### train된 모델을 가지고 validation set을 input하여 prediction을 그려서 모델이 학습하는지 확인하는 부분 ###########
# from detectron2.engine import DefaultPredictor
# from detectron2.utils.visualizer import Visualizer
#
# cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
# cfg.DATASETS.TEST = (['ADDxywht_val'])
# predictor = DefaultPredictor(cfg)
# dataset_dicts = get_ADDxywht_dicts(dataset_dir,'val')
# ADDxywht_metadata = MetadataCatalog.get("ADDxywht_val")
# for d in random.sample(dataset_dicts, 3):
#     im = cv2.imread(d["file_name"])
#     outputs = predictor(im)
#     v = Visualizer(im[:, :, ::-1],
#                    metadata=ADDxywht_metadata,
#                    scale=0.3
#     )
#     v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
#     cv2.imshow('test',v.get_image()[:, :, ::-1])
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
##################################################################








