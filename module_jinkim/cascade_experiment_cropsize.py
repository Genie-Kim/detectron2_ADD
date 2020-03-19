# -*- coding: utf-8 -*-
from detectron2.evaluation import (
    DatasetEvaluator,
    inference_on_dataset,
    print_csv_format,
    verify_results,
)
from module_jinkim import ADDDatasetMapper, ADDDatasetMapperTEST
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.evaluation import inference_on_dataset, RotatedCOCOEvaluator, DatasetEvaluators
from detectron2 import model_zoo
from detectron2.data import transforms as T
from detectron2.data import detection_utils as utils
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader, build_detection_train_loader
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.layers import ShapeSpec, batched_nms_rotated
from detectron2.structures import Instances, RotatedBoxes, pairwise_iou_rotated
import os, torch, copy, random, cv2, math, pdb
import numpy as np
import pandas as pd
from tqdm import tqdm

from detectron2.structures import BoxMode

pd.set_option('display.max_columns', 30)
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


def get_ADDxywht_traindicts(dataset_dir):
    csv_file = os.path.join(dataset_dir, "labels.csv")
    img_dir = os.path.join(dataset_dir, 'images')
    df = pd.read_csv(csv_file)
    image_shape = 3000

    # XYWHA 데이터 생성하기
    xywha = backward_convert(df.loc[:, 'point1_x':'point4_y'].values)
    sub_df = pd.DataFrame(data=xywha, columns=np.array(['center_x', 'center_y', 'width', 'height', 'angle']))
    df = df.join(sub_df)
    df['angle'] = -1 * df['angle']  # cv2의 minarearect랑 표현 방식이 다르다. detectron2는 ccw라고 함. 근데 -1을 곱해야 맞음

    df = (df.reset_index()).drop(columns='index')  # annotation이 몇개 빠지기 때문에 reset index 해준다.

    file_name_dict = {}
    for i in tqdm(range(df.shape[0]),desc="get ADD train dataset upload"):
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

        file_name_dict[filename]["annotations"].append(obj)

    dataset_dicts = []
    for filename in file_name_dict.keys():
        record = file_name_dict[filename]
        record['file_name'] = filename
        dataset_dicts.append(record)
    return dataset_dicts

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

def get_ADDtest_dicts(test_dataset_dir,crop_size):
    img_dir = os.path.join(test_dataset_dir,str(crop_size)+'_cropped_images')

    image_shape = crop_size

    file_name_dict = {}
    for file_name in tqdm(os.listdir(img_dir),desc = 'inference data crop size = {0} dictionary uploading'.format(crop_size)):
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


############################## config file setting #################################################
ClassCount = 4
# input_image_scale = 550 # ADD dataset crop 안했을 때 image scale
image_resize_size = 750  # 이 코드에서 maximum 인풋이미지의 사이즈는 이것으로 결정된다.
model_to_resume = 'model_0011249.pth'

cfg = get_cfg()
cfg.OUTPUT_DIR = './module_jinkim/output'
configfile = "Misc/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv.yaml"
cfg.merge_from_file(model_zoo.get_config_file(configfile))
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, model_to_resume)

cfg.DATASETS.TEST =  (['ADDxywht_test']) # 상관 없음.
# Size of the smallest side of the image during testing. Set to zero to disable resize in testing.
cfg.INPUT.MIN_SIZE_TEST = image_resize_size
# Maximum size of the side of the image during testing
cfg.INPUT.MAX_SIZE_TEST = image_resize_size

cfg.DATALOADER.NUM_WORKERS = 4

cfg.MODEL.MASK_ON = False
cfg.MODEL.ROI_HEADS.NAME = "CascadeRROIHeads" # cascade roi heads가 rroi heads를 상속받아서 구현된 것.
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512  # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = ClassCount
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model

cfg.MODEL.PROPOSAL_GENERATOR.NAME = "RRPN"

cfg.MODEL.RPN.BBOX_REG_WEIGHTS = (1, 1, 1, 1, 1)
anrsc_p_pmd = [1,2**(1/3)] # anchor scales per pyramid
cfg.MODEL.ANCHOR_GENERATOR.NAME = "RotatedAnchorGenerator"
cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[32*x for x in anrsc_p_pmd], [64*x for x in anrsc_p_pmd], [128*x for x in anrsc_p_pmd], [256*x for x in anrsc_p_pmd], [512*x for x in anrsc_p_pmd]]
cfg.MODEL.ANCHOR_GENERATOR.ANGLES = [[15, 30, 45, 60,75, 90]]
cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.2, 1/3, 0.5, 2.0, 3, 5]]  # 수정

cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE = "ROIAlignRotated"
cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS = (10.0, 10.0, 5.0, 5.0, 10.0)

# The number of cascade stages is implicitly defined by the length of the following two configs.
cfg.MODEL.ROI_BOX_CASCADE_HEAD.BBOX_REG_WEIGHTS = (
    (10.0, 10.0, 5.0, 5.0, 10.0),
    (20.0, 20.0, 10.0, 10.0, 20.0),
    (30.0, 30.0, 15.0, 15.0, 30.0),
)

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
########### dataset setting part ###########
from detectron2.data import DatasetCatalog, MetadataCatalog

train_inference_file_name = "ADDxywht_train650"
cropped_dataset_dir = os.path.expanduser('~/hddu/ADD_all_original_data/')  # symbolik link를 유저 path 바로 밑에 설치.(ex :sudo ln -sT ~/hddu/dataset_ADD_20191122/ ~/ADD_dataset)
train_dataset_dir = '/home/genie/hddu/ADD_all_original_data/ADD_12merged_original/'

DatasetCatalog.register(train_inference_file_name, lambda d=1: get_ADDxywht_traindicts(train_dataset_dir))
MetadataCatalog.get(train_inference_file_name).set(thing_classes=['container', 'oil tanker', 'aircraft carrier', 'maritime vessels'])

for crop_size in [550,650,750,850,950,1050,1150,1250,1350,1450,1550]:
    DatasetCatalog.register("ADDxywht_test_"+str(crop_size), lambda crop_size=crop_size: get_ADDtest_dicts(cropped_dataset_dir,crop_size))
    MetadataCatalog.get("ADDxywht_test_"+str(crop_size)).set(thing_classes=['container', 'oil tanker', 'aircraft carrier', 'maritime vessels'])


##################################################################
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
        return build_detection_test_loader(cfg, dataset_name, mapper=ADDDatasetMapperTEST(cfg, False,False))

########### test label 뽑아서 csv로 만드는 과정 ###########
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer

cropsize_to_infer = [650]
crop_namimg = '_'.join([str(x) for x in cropsize_to_infer])
predictor = DefaultPredictor(cfg)
write_csv_path = os.path.join(cfg.OUTPUT_DIR,'ADD_crop_exp_result_'+crop_namimg+'_'+os.path.splitext(model_to_resume)[0]+'.csv')
all_classes = ['container', 'oil tanker', 'aircraft carrier', 'maritime vessels']

pred_alltest=[]

infer_datasets = {}
for crop_size in cropsize_to_infer:
    infer_datasets[crop_size] = get_ADDtest_dicts(cropped_dataset_dir,crop_size)

data_loader = MyTrainer.build_test_loader(cfg, train_inference_file_name)
evaluator = MyTrainer.build_evaluator(cfg, train_inference_file_name)
evaluator.reset()


for idx, inputs in enumerate(tqdm(data_loader,desc='train set images evaluating')):
    # 하나의 ndarray form으로 만들자.
    npboxes = np.array([])
    npscores = np.array([])
    npclasses = np.array([])

    for crop_size in infer_datasets.keys():

        original_filename = os.path.splitext(os.path.basename(inputs[0]['file_name']))[0] # .png도 빠진 오리지날 트레이닝 셋의 번
        infer_images_path = []
        for infer_image in infer_datasets[crop_size]:
            infer_filename = os.path.splitext(os.path.basename(infer_image['file_name']))[0]
            if infer_filename.split('_')[0] == original_filename:
                infer_images_path.append(infer_image['file_name'])
        infer_prediction_percropsizeimg =[]

        for infer_path in infer_images_path:
            im = cv2.imread(infer_path)
            outputs = predictor(im) # output is cropped size version.
            predictions = outputs["instances"].to("cpu")
            boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
            scores = predictions.scores if predictions.has("scores") else None
            classes = predictions.pred_classes if predictions.has("pred_classes") else None

            infer_filename = os.path.splitext(os.path.basename(infer_path))[0]
            boxes = boxes.tensor.numpy()
            scores = scores.numpy()
            classes = classes.numpy()

            ## revert box to original images resolution.
            infer_image_start_coord = infer_filename.split('_')[1:3]
            boxes[:,0] = boxes[:,0]+int(infer_image_start_coord[1])
            boxes[:,1] = boxes[:,1]+int(infer_image_start_coord[0])
            ########################################################
            obj = {}
            obj['boxes'] = boxes # numpy array
            obj['scores'] = scores
            obj['classes'] = classes
            infer_prediction_percropsizeimg.append(obj)

        for inference in infer_prediction_percropsizeimg:
            if inference['boxes'].shape[0] != 0:
                if npboxes.shape[0] == 0 and npscores.shape[0] == 0 and npclasses.shape[0] == 0:
                    npboxes = inference['boxes']
                    npscores = inference['scores']
                    npclasses = inference['classes']
                elif npboxes.shape[0]!=npscores.shape[0] or npboxes.shape[0]!=npclasses.shape[0] or npclasses.shape[0]!=npscores.shape[0]:
                    print('box score classess shape is not same')
                    os.abort()
                else:
                    npboxes = np.append(npboxes,inference['boxes'],axis=0)
                    npscores = np.append(npscores,inference['scores'],axis=0)
                    npclasses = np.append(npclasses,inference['classes'],axis=0)
    # perform NMS
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    boxes = torch.from_numpy(npboxes).float().to(device)
    scores = torch.from_numpy(npscores).float().to(device)
    classes = torch.from_numpy(npclasses).float().to(device)
    # classes = classes.type(torch.int64)
    nms_thresh = 0.5
    if boxes.shape[0]!=0:
        keep = batched_nms_rotated(boxes, scores, classes, nms_thresh)
        topk_per_image = 100
        keep = keep[:topk_per_image]
        boxes, scores, classes = boxes[keep], scores[keep], classes[keep]
    # evaluate
    result = Instances((3000, 3000))
    result.pred_boxes = RotatedBoxes(boxes)
    result.scores = scores
    result.pred_classes = classes
    outputs = []
    output_dict = {}
    output_dict['instances'] = result
    outputs.append(output_dict)
    evaluator.process(inputs, outputs)

    # extract for writing csv file
    # csv 저장.
    if boxes.shape[0] != 0:
        boxes[:, 4] = boxes[:, 4] * -1  # cv2와 detectron은 angle 방향이 다른것 같음. 확인해볼 필요 있음.
        cord_xy4 = forward_convert(boxes)
        scores = scores.cpu().numpy()
        classes = classes.cpu().numpy()
        for i in range(cord_xy4.shape[0]):
            rbbox = cord_xy4[i].tolist()
            conf = scores[i]
            class_id = int(classes[i] + 1)
            label = all_classes[int(classes[i])]
            filename = os.path.basename(inputs[0]['file_name'])
            one_obj_anno = [filename, class_id, conf] + rbbox
            pred_alltest.append(one_obj_anno)

results = evaluator.evaluate()
if results is None:
    results = {}
print_csv_format(results)

temp_df = pd.DataFrame(data= np.asarray(pred_alltest), columns = np.array(['file_name','class_id','confidence', 'point1_x', 'point1_y', 'point2_x', 'point2_y','point3_x', 'point3_y', 'point4_x', 'point4_y']))
temp_df.to_csv(write_csv_path, mode='w', index=False)
filename_wts = '0.png'
draw_gts_onimage_save(temp_df,filename_wts,os.path.join(train_dataset_dir,'images'),os.path.join(cfg.OUTPUT_DIR,'draw_pred_cropexp_'+crop_namimg+filename_wts))



    # v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    # cv2.imshow('test',v.get_image()[:, :, ::-1])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
##################################################################








