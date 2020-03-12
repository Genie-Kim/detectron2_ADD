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

pd.set_option('display.max_columns', 30)

def draw_gts_onimage_save(add_df,file_name,img_path,save_path,thickness = 10):
    # image save code, add_df 는 add dataset csv 라벨을 pandas dataframe으로 올린것.
    # (ac_df.sort_index(by=['area']))[['file_name','area']] <--  ac부분만 원하는걸로 바꿔서 scale별로 소팅가
    # Ex : draw_gts_onimage_save(add_df,'285.png','/home/genie/hddu/dataset_ADD_20191122/train/images','/home/genie/hddu/dataset_ADD_20191122/train/analysis/aircraft_carrier')
    img_file_path = os.path.join(img_path,file_name)
    img = cv2.imread(img_file_path)
    one_file_df = add_df[add_df['file_name'] == file_name]

    for i in range(one_file_df.shape[0]):
        one_obj = one_file_df.iloc[i,:]
        class_id = int(one_obj['class_id'])

        if class_id == 1:
            poly_color = (255, 0, 0) # blue CONTAINER
        if class_id == 2:
            poly_color = (0, 255, 0) # green OIL TANKER
        if class_id == 3:
            poly_color = (0, 0, 255) # red AIR CRAFT
        if class_id == 4:
            poly_color = (255, 255, 0) # cyon : MARITIM VESSEL

        pts_array = (one_obj.loc['point1_x':'point4_y']).values
        pts_array = [float(x) for x in pts_array]
        pts = np.array([[pts_array[0], pts_array[1]], [pts_array[2], pts_array[3]],
                        [pts_array[4], pts_array[5]], [pts_array[6], pts_array[7]]])

        img = cv2.polylines(img, np.int32([pts]), True, poly_color, thickness)
    cv2.imwrite(save_path, img)


def forward_convert(coordinate, with_label=False):
    """
    :param coordinate: format [x_c, y_c, w, h, theta]
    :return: format [x1, y1, x2, y2, x3, y3, x4, y4]
    """

    boxes = []
    if with_label:
        for rect in coordinate:
            box = cv2.boxPoints(((rect[0], rect[1]), (rect[2], rect[3]), rect[4]))
            box = np.reshape(box, [-1, ])
            boxes.append([box[0], box[1], box[2], box[3], box[4], box[5], box[6], box[7], rect[5]])
    else:
        for rect in coordinate:
            box = cv2.boxPoints(((rect[0], rect[1]), (rect[2], rect[3]), rect[4]))
            box = np.reshape(box, [-1, ])
            boxes.append([box[0], box[1], box[2], box[3], box[4], box[5], box[6], box[7]])

    return np.array(boxes, dtype=np.float32)

def get_ADDtest_dicts(test_dataset_dir):
    img_dir = os.path.join(test_dataset_dir, 'images')

    image_shape = input_image_scale

    file_name_dict = {}
    for file_name in os.listdir(img_dir):
        filename = os.path.join(img_dir,file_name)
        height = image_shape
        width = image_shape

        if filename not in file_name_dict:
            file_name_dict[filename] = dict.fromkeys(["image_id", "height", "width"])
            image_id = len(file_name_dict)
            file_name_dict[filename]["image_id"] = image_id
            file_name_dict[filename]["height"] = height
            file_name_dict[filename]["width"] = width

    dataset_dicts = []
    for filename in file_name_dict.keys():
        record = file_name_dict[filename]
        record['file_name'] = filename
        dataset_dicts.append(record)
    return dataset_dicts


########### dataset setting part ###########
from detectron2.data import DatasetCatalog, MetadataCatalog

dataset_dir = os.path.expanduser('~/ADD_dataset/test/')  # symbolik link를 유저 path 바로 밑에 설치.(ex :sudo ln -sT ~/hddu/dataset_ADD_20191122/ ~/ADD_dataset)


DatasetCatalog.register("ADDxywht_test", lambda d = 1: get_ADDtest_dicts(dataset_dir))
MetadataCatalog.get("ADDxywht_test").set(thing_classes=['container', 'oil tanker', 'aircraft carrier', 'maritime vessels'])
ADDxywht_metadata = MetadataCatalog.get("ADDxywht_test")
##################################################################

############################## config file setting #################################################
ClassCount = 4
input_image_scale = 3000 # ADD dataset crop 안했을 때 image scale
max_image_resize = 1250  # 이 코드에서 maximum 인풋이미지의 사이즈는 이것으로 결정된다.
model_to_resume = 'model_0040299.pth'

cfg = get_cfg()
cfg.OUTPUT_DIR = './module_jinkim/output'
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, model_to_resume)

cfg.DATASETS.TEST = (['ADDxywht_test'])
# Size of the smallest side of the image during testing. Set to zero to disable resize in testing.
cfg.INPUT.MIN_SIZE_TEST = max_image_resize
# Maximum size of the side of the image during testing
cfg.INPUT.MAX_SIZE_TEST = max_image_resize

cfg.DATALOADER.NUM_WORKERS = 4

cfg.MODEL.MASK_ON = False
cfg.MODEL.ROI_HEADS.NAME = "RROIHeads"
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512  # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = ClassCount
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
cfg.MODEL.PROPOSAL_GENERATOR.NAME = "RRPN"
cfg.MODEL.RPN.HEAD_NAME = "StandardRPNHead"
cfg.MODEL.RPN.BBOX_REG_WEIGHTS = (1, 1, 1, 1, 1)
cfg.MODEL.ANCHOR_GENERATOR.NAME = "RotatedAnchorGenerator"
cfg.MODEL.ANCHOR_GENERATOR.ANGLES = [[0, 30, 60, 90]]
# cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.0, 2.0]] # 원본
cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.2, 0.2857, 0.5, 2.0, 3.5, 5]]  # 수정

cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE = "ROIAlignRotated"
cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS = (10.0, 10.0, 5.0, 5.0, 10.0)

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
print('FINAL CONFIG FILE')
print(cfg)
########################################################################################################

########### test label 뽑아서 csv로 만드는 과정 ###########
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer

predictor = DefaultPredictor(cfg)
dataset_dicts = get_ADDtest_dicts(dataset_dir)
write_csv_path = os.path.join(cfg.OUTPUT_DIR,'ADD_test_result_'+model_to_resume+'.csv')
pred_alltest = []

for one_image in tqdm(dataset_dicts,desc='CALCULATING PREDICTION for TEST DATASET...'):
    im = cv2.imread(one_image["file_name"])
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                   metadata=ADDxywht_metadata,
                   scale=1
    )
    predictions = outputs["instances"].to("cpu")
    boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
    scores = predictions.scores if predictions.has("scores") else None
    classes = predictions.pred_classes if predictions.has("pred_classes") else None
    boxes = boxes.tensor.numpy()
    boxes[:, 4] = boxes[:, 4] * -1 # cv2와 detectron은 angle 방향이 다른것 같음. 확인해볼 필요 있음.
    cord_xy4 = forward_convert(boxes)
    scores = scores.numpy()
    classes = classes.numpy()
    all_classes = v.metadata.get("thing_classes", None)

    for i in range(len(predictions)):
        rbbox = cord_xy4[i].tolist()
        conf = scores[i]
        class_id= classes[i] + 1
        label = all_classes[classes[i]]
        filename = os.path.basename(one_image["file_name"])
        one_obj_anno = [filename, class_id, conf ] + rbbox
        pred_alltest.append(one_obj_anno)

temp_df = pd.DataFrame(data= np.asarray(pred_alltest), columns = np.array(['file_name','class_id','confidence', 'point1_x', 'point1_y', 'point2_x', 'point2_y','point3_x', 'point3_y', 'point4_x', 'point4_y']))
temp_df.to_csv(write_csv_path, mode='w', index=False)
filename_wts = '47.png'
draw_gts_onimage_save(temp_df,filename_wts,'/home/genie/hddu/dataset_ADD_20191122/test/images',os.path.join(cfg.OUTPUT_DIR,'draw_pred_'+filename_wts))



    # v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    # cv2.imshow('test',v.get_image()[:, :, ::-1])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
##################################################################








