import multiprocessing
import os
import cv2
from tqdm import tqdm


def extract_label_per_img(df, img_name):
    format_data = []
    img_df = df[df['file_name']==img_name]
    label = img_df.loc[:, 'point1_x':'point4_y']
    label['class_num'] = img_df['class_id']
    return label.values

def clip_image(file_idx, image, width, height, stride_w, stride_h,cropped_img_savepath,padding_size=0,value=0):
        shape = image.shape
        # padding_size 는 전체 이미지에 대한 비율 ex) 1/6 ,1/7, 1/8
        if padding_size > 0:
            padding_size = shape*padding_size
            image = cv2.copyMakeBorder(image, padding_size, padding_size, padding_size, padding_size, cv2.BORDER_CONSTANT,value)
            # cropping 할 때 padding을 넣는게 좋을 까? 라는 생각을 함.(자르기 전에 초기에 넣자).

        for start_h in range(0, shape[0], stride_h):
            for start_w in range(0, shape[1], stride_w):

                start_h_new = start_h
                start_w_new = start_w
                if start_h + height > shape[0]:
                    start_h_new = shape[0] - height
                if start_w + width > shape[1]:
                    start_w_new = shape[1] - width
                top_left_row = max(start_h_new, 0)
                top_left_col = max(start_w_new, 0)
                bottom_right_row = min(start_h + height, shape[0])
                bottom_right_col = min(start_w + width, shape[1])

                subImage = image[top_left_row:bottom_right_row, top_left_col: bottom_right_col]


                if (subImage.shape[0] > 5 and subImage.shape[1] > 5):
                    crop_img_name = "%s_%04d_%04d_%04d.png" % (file_idx, top_left_row, top_left_col,width)
                    crop_img_path = os.path.join(cropped_img_savepath,crop_img_name)
                    cv2.imwrite(crop_img_path, subImage)


def cropImg_makeLabel_Save_multiproc(crop_size):
    padding_size = 0
    save_folder = os.path.join(test_folder, str(crop_size) + '_cropped_images/')
    img_h, img_w = crop_size, crop_size
    stride_h = int(crop_size*4/5)
    stride_w = int(crop_size*4/5)
    os.makedirs(save_folder, exist_ok=True)
    images = [i for i in os.listdir(IMAGE_PATH1) if 'png' in i]
    for idx, img in enumerate(images):
        print('shape {0} cropping rate : {1}'.format(img_w,idx/len(images)))
        img_data = cv2.imread(os.path.join(IMAGE_PATH1, img))
        clip_image(img.strip('.png'), img_data, img_w, img_h, stride_w, stride_h,save_folder,padding_size)  # crop & gt refine & saving

    print('-*--*--*--*-shape {0} cropping DONE-*--*--*--*-'.format(img_w))


test_folder = os.path.expanduser('~/ADD_dataset/test') # 테스트 이미지들의 위치
IMAGE_PATH1 = os.path.expanduser('~/ADD_dataset/test/images') # 테스트 이미지를 저장할

if __name__ == "__main__":

    # for multi processing
    cropping_size_list = [550,750,1050,1250,1550]

    pool = multiprocessing.Pool(processes=5) # io 이슈 때문에 5개가 한계이다...
    pool.map(cropImg_makeLabel_Save_multiproc,cropping_size_list)
    pool.close()
    pool.join()
    print('processing done')






