#提案手法１ソース

import sys
import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
from keras import backend as K
from keras.preprocessing import image
from keras.models import Sequential, Model
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.layers import Input, Activation, Dropout, Flatten, Dense
from PIL import Image
from model import Deeplabv3


import tensorflow as tf
from tensorflow.python.framework import ops



result_dir = 'results'

#numpy配列の省略を解除するコマンド
#np.set_printoptions(threshold=np.inf)

#deeplabv3+,モデル読み込み
deeplab_model = Deeplabv3()

classes = ['001.Black_footed_Albatross','002.Laysan_Albatross','003.Sooty_Albatross','004.Groove_billed_Ani',
            '005.Crested_Auklet','006.Least_Auklet','007.Parakeet_Auklet','008.Rhinoceros_Auklet',
            '009.Brewer_Blackbird','010.Red_winged_Blackbird','011.Rusty_Blackbird','012.Yellow_headed_Blackbird',
            '013.Bobolink','014.Indigo_Bunting','015.Lazuli_Bunting','016.Painted_Bunting','017.Cardinal',
            '018.Spotted_Catbird','019.Gray_Catbird','020.Yellow_breasted_Chat','021.Eastern_Towhee',
            '022.Chuck_will_Widow','023.Brandt_Cormorant','024.Red_faced_Cormorant','025.Pelagic_Cormorant',
            '026.Bronzed_Cowbird','027.Shiny_Cowbird','028.Brown_Creeper','029.American_Crow','030.Fish_Crow',
            '031.Black_billed_Cuckoo','032.Mangrove_Cuckoo','033.Yellow_billed_Cuckoo','034.Gray_crowned_Rosy_Finch',
            '035.Purple_Finch','036.Northern_Flicker','037.Acadian_Flycatcher','038.Great_Crested_Flycatcher',
            '039.Least_Flycatcher','040.Olive_sided_Flycatcher','041.Scissor_tailed_Flycatcher','042.Vermilion_Flycatcher',
            '043.Yellow_bellied_Flycatcher','044.Frigatebird','045.Northern_Fulmar','046.Gadwall','047.American_Goldfinch',
            '048.European_Goldfinch','049.Boat_tailed_Grackle','050.Eared_Grebe','051.Horned_Grebe',
            '052.Pied_billed_Grebe','053.Western_Grebe','054.Blue_Grosbeak','055.Evening_Grosbeak',
            '056.Pine_Grosbeak','057.Rose_breasted_Grosbeak','058.Pigeon_Guillemot','059.California_Gull',
            '060.Glaucous_winged_Gull','061.Heermann_Gull','062.Herring_Gull','063.Ivory_Gull','064.Ring_billed_Gull',
            '065.Slaty_backed_Gull','066.Western_Gull','067.Anna_Hummingbird','068.Ruby_throated_Hummingbird',
            '069.Rufous_Hummingbird','070.Green_Violetear','071.Long_tailed_Jaeger','072.Pomarine_Jaeger',
            '073.Blue_Jay','074.Florida_Jay','075.Green_Jay','076.Dark_eyed_Junco','077.Tropical_Kingbird',
            '078.Gray_Kingbird','079.Belted_Kingfisher','080.Green_Kingfisher','081.Pied_Kingfisher','082.Ringed_Kingfisher',
            '083.White_breasted_Kingfisher','084.Red_legged_Kittiwake','085.Horned_Lark','086.Pacific_Loon',
            '087.Mallard','088.Western_Meadowlark','089.Hooded_Merganser','090.Red_breasted_Merganser','091.Mockingbird',
            '092.Nighthawk','093.Clark_Nutcracker','094.White_breasted_Nuthatch','095.Baltimore_Oriole',
            '096.Hooded_Oriole','097.Orchard_Oriole','098.Scott_Oriole','099.Ovenbird','100.Brown_Pelican',
            '101.White_Pelican','102.Western_Wood_Pewee','103.Sayornis','104.American_Pipit','105.Whip_poor_Will',
            '106.Horned_Puffin','107.Common_Raven','108.White_necked_Raven','109.American_Redstart',
            '110.Geococcyx','111.Loggerhead_Shrike','112.Great_Grey_Shrike','113.Baird_Sparrow',
            '114.Black_throated_Sparrow','115.Brewer_Sparrow','116.Chipping_Sparrow','117.Clay_colored_Sparrow',
            '118.House_Sparrow','119.Field_Sparrow','120.Fox_Sparrow','121.Grasshopper_Sparrow','122.Harris_Sparrow',
            '123.Henslow_Sparrow','124.Le_Conte_Sparrow','125.Lincoln_Sparrow','126.Nelson_Sharp_tailed_Sparrow',
            '127.Savannah_Sparrow','128.Seaside_Sparrow','129.Song_Sparrow','130.Tree_Sparrow','131.Vesper_Sparrow',
            '132.White_crowned_Sparrow','133.White_throated_Sparrow','134.Cape_Glossy_Starling','135.Bank_Swallow',
            '136.Barn_Swallow','137.Cliff_Swallow','138.Tree_Swallow','139.Scarlet_Tanager','140.Summer_Tanager',
            '141.Artic_Tern','142.Black_Tern','143.Caspian_Tern','144.Common_Tern','145.Elegant_Tern',
            '146.Forsters_Tern','147.Least_Tern','148.Green_tailed_Towhee','149.Brown_Thrasher',
            '150.Sage_Thrasher','151.Black_capped_Vireo','152.Blue_headed_Vireo','153.Philadelphia_Vireo','154.Red_eyed_Vireo',
            '155.Warbling_Vireo','156.White_eyed_Vireo','157.Yellow_throated_Vireo','158.Bay_breasted_Warbler',
            '159.Black_and_white_Warbler','160.Black_throated_Blue_Warbler','161.Blue_winged_Warbler','162.Canada_Warbler',
            '163.Cape_May_Warbler','164.Cerulean_Warbler','165.Chestnut_sided_Warbler','166.Golden_winged_Warbler',
            '167.Hooded_Warbler','168.Kentucky_Warbler','169.Magnolia_Warbler','170.Mourning_Warbler','171.Myrtle_Warbler',
            '172.Nashville_Warbler','173.Orange_crowned_Warbler','174.Palm_Warbler','175.Pine_Warbler',
            '176.Prairie_Warbler','177.Prothonotary_Warbler','178.Swainson_Warbler','179.Tennessee_Warbler',
            '180.Wilson_Warbler','181.Worm_eating_Warbler','182.Yellow_Warbler','183.Northern_Waterthrush',
            '184.Louisiana_Waterthrush','185.Bohemian_Waxwing','186.Cedar_Waxwing','187.American_Three_toed_Woodpecker',
            '188.Pileated_Woodpecker','189.Red_bellied_Woodpecker','190.Red_cockaded_Woodpecker',
            '191.Red_headed_Woodpecker','192.Downy_Woodpecker','193.Bewick_Wren','194.Cactus_Wren',
            '195.Carolina_Wren','196.House_Wren','197.Marsh_Wren','198.Rock_Wren','199.Winter_Wren','200.Common_Yellowthroat']

# Define model here ---------------------------------------------------
def build_model():
    """Function returning keras model instance.
    
    Model can be
     - Trained here
     - Loaded with load_model
     - Loaded from keras.applications
    """
    #VGG16 model load
    input_tensor = Input(shape=(150, 150, 3))
    vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)

    # FC
    fc = Sequential()
    fc.add(Flatten(input_shape=vgg16.output_shape[1:]))
    fc.add(Dense(256, activation='relu'))
    fc.add(Dropout(0.5))
    fc.add(Dense(200, activation='softmax'))

    # VGG16とFCを接続
    model = Model(input=vgg16.input, output=fc(vgg16.output))

    # 学習済みの重みをロード
    model.load_weights(os.path.join(result_dir, 'step320_finetuning.h5'))
    
    return model 

H, W = 150, 150 # Input shape, defined by the model (model.input_shape)

# ---------------------------------------------------------------------

#load image here ------------------------------------------------------
def load_image(path, preprocess=True):
    """Load and preprocess image."""
    x = image.load_img(path, target_size=(H, W))
    if preprocess:
        x = image.img_to_array(x)
        x = np.expand_dims(x, axis=0)
        x = x / 255.0
    return x
#----------------------------------------------------------------------


#guided_backprop, guided_cam用, 本研究では無関係-------------------------------
def deprocess_image(x):
    """Same normalization as in:
    https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py
    """
    x = x.copy()
    if np.ndim(x) > 3:
        x = np.squeeze(x)
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_dim_ordering() == 'th':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x
#------------------------------------------------------------------------------------

#normalize----------------------------------------------------
def normalize(x):
    """Utility function to normalize a tensor by its L2 norm"""
    return (x + 1e-10) / (K.sqrt(K.mean(K.square(x))) + 1e-10)
#----------------------------------------------------------------


#define guided_cam model here 研究では未使用-----------------------------------------
def build_guided_model():
    """Function returning modified model.
    
    Changes gradient function for all ReLu activations
    according to Guided Backpropagation.
    """
    if "GuidedBackProp" not in ops._gradient_registry._registry:
        @ops.RegisterGradient("GuidedBackProp")
        def _GuidedBackProp(op, grad):
            dtype = op.inputs[0].dtype
            return grad * tf.cast(grad > 0., dtype) * \
                   tf.cast(op.inputs[0] > 0., dtype)

    g = tf.get_default_graph()
    with g.gradient_override_map({'Relu': 'GuidedBackProp'}):
        new_model = build_model()
    return new_model
#------------------------------------------------------------------------------------



#guided_backprop用, 研究では未使用--------------------------------------------------
def guided_backprop(input_model, images, layer_name):
    """Guided Backpropagation method for visualizing input saliency."""
    input_imgs = input_model.input
    layer_output = input_model.get_layer(layer_name).output
    grads = K.gradients(layer_output, input_imgs)[0]
    backprop_fn = K.function([input_imgs, K.learning_phase()], [grads])
    grads_val = backprop_fn([images, 0])[0]
    return grads_val
#-----------------------------------------------------------------------------------



#grad_cam program-----------------------------------------------------------------------
def grad_cam(input_model, image, cls, layer_name):
    """GradCAM metラベリングod for visualizing input saliency."""

    #モデルの出力ラベリングンソルを取得
    y_c = input_model.output[0, cls]
   
    #指定した層のラベリング力テンソルを取得
    conv_output = input_model.get_layer(layer_name).output
    
    #可視化したい層の出力の、モデルの出力に関しての勾配を取得　　　
    grads = K.gradients(y_c, conv_output)[0]
   
    # Normalize if necessary
    # grads = normalize(grads)
    gradient_function = K.function([input_model.input], [conv_output, grads])
    
    #画像の4次元テンソルを入力、可視化したい層の出力テンソル及びモデルの出力との勾配を取得 
    output, grads_val = gradient_function([image])

    output, grads_val = output[0, :], grads_val[0, :, :, :]
    
    #平均を取ってgrads_valを1次元に削減(要素数512)
    weights = np.mean(grads_val, axis=(0, 1))
    
    #outputsとgrads_valの内積を取る(結果は2次元配列)
    cam = np.dot(output, weights)

    # Process CAM
    #画像サイズに合わせ150*150に拡大
    cam = cv2.resize(cam, (H, W), cv2.INTER_LINEAR)
    
    #camの要素が0より大きいならそのまま、小さければ0に置き換える
    cam = np.maximum(cam, 0)
    
    #正規化
    cam = cam / cam.max()

    return cam
#-----------------------------------------------------------------------------------------------------



#classfication and visuallize here ----------------------------------------------------------------------------------
def compute_saliency(model, guided_model, img_path, layer_name='block5_conv3', cls=-1, visualize=True, save=True):
    """Compute saliency using all three approaches.
        -layer_name: layer to compute gradients;
        -cls: class number to localize (-1 for most probable class).
    """
    data_dir_path = img_path

    rank_score = 0
    rank_score_neo = 0
    

    dir_path = img_path.split("/")
    dir_name = dir_path[1].split(".")


    file_list = os.listdir(img_path)

    for file_name in file_list:

        #get image name
        abs_name = data_dir_path + file_name

        #grad_cam用元画像ロード
        preprocessed_input = load_image(abs_name)

        # ----------------- Semantic Segmentation 処理 ---------------------------
        #semantic segmentation用元画像ロード
        imgpre = Image.open(abs_name) 
        img_resize = imgpre.resize((400,500))
        img_resize.save("resize/resize_image.jpg")
        
        #segmentation用にpyplotでロード
        img = plt.imread("resize/resize_image.jpg")
        #pyplot画像のw,h,channelを取得
        h, w, _ = img.shape

        #pyplot画像の最大幅、最小幅の入力サイズ512に対する比率を取得
        ratio = 512. / np.max([w,h])

        #pyplot画像サイズを512になるように調整
        resized = cv2.resize(img,(int(ratio*h),int(ratio*w)))
        
        #RGB値を -1 〜 1 に変換
        resized = resized / 127.5 - 1.

        #wが最大幅なら0を格納、hが最大幅なら512 - w の値が入る
        pad_x = int(512 - resized.shape[0])

        #512*512にするために画素が足りないところをゼロパティング
        resized2 = np.pad(resized,((0,pad_x),(0,0),(0,0)),mode='constant')
        
        #segmentationのpredict
        res = deeplab_model.predict(np.expand_dims(resized2,0))

        #ラベルを取得
        labels = np.argmax(res.squeeze(),-1)

        #sementation resultを保存
        plt.imshow(labels[:-pad_x])
        plt.savefig("segmentation/result.jpg",transparent=True)

        plt.close()

        #サイズ調整
        image = Image.open("segmentation/result.jpg")
        crop_im = image.crop((105,65,555,420))
        crop_im.save("crop_image/crop_image.jpg")

        #grad_cam画像と合わせるため150*150にリサイズ
        crop_resize = Image.open("crop_image/crop_image.jpg")
        resize_image = crop_resize.resize((150,150))
        resize_image.save("resize/crop_resize.jpg")

        #RGBを読み取るのでopencvで読み込む
        seg_img = cv2.imread("resize/crop_resize.jpg")

        #背景、物体部分でラベリング
        object_list = []

        for i in range(150):
            for m in range(150):
                #鳥部分である場合
                if seg_img[i][m,2] >= 220 or seg_img[i][m,0] >= 110:
                    object_list.append([m,i])
        
        object_index = np.array(object_list)
        #-------------------------------------------------------------------------------


        #後にcutする元画像
        original_im = Image.open(abs_name)
        #元画像のサイズを取得
        ori_w,ori_h = original_im.size
        

        #------------------------- grad_cam処理 -------------------------------------

        #元画像でclassfication
        predictions = model.predict(preprocessed_input)
        top_n = 200
        pred = model.predict(preprocessed_input)[0]
        top_indices = pred.argsort()[-top_n:][::-1]
        result = [(classes[i], pred[i]) for i in top_indices]
        rank_count = 0

        print(file_name)
        print("\n")

        print("original_result")

        #classfication result
        for x in result:
            rank_count += 1

            #分類したクラスのインデックス取得
            class_name,rlt = x
            cls_index = class_name.split(".")

            #適合率算出
            if dir_name[0] == cls_index[0]:
                rank_score += 1 / rank_count
                score = 1 / rank_count
                print("\n")
                print("rank")
                print(rank_count)
                print("\n")
                print("score")
                print(score)

        print("\n")

        #分類結果１位のインデックス取得
        if cls == -1:
            cls = np.argmax(predictions)
            
        #visuallize
        gradcam = grad_cam(model, preprocessed_input, cls, layer_name)
        gb = guided_backprop(guided_model, preprocessed_input, layer_name)
        guided_gradcam = gb * gradcam[..., np.newaxis]

        if save:
            #camの値を元に1画素ごとに[B,G,R]の順で0~255の値を格納
            jetcam = cv2.applyColorMap(np.uint8(255 * gradcam), cv2.COLORMAP_JET)
            jetcam = (np.float32(jetcam) + load_image(abs_name, preprocess=False)) / 2

            root,ext = os.path.splitext(file_name)

            i_min = 0
            i_max = 0
            m_min = 0
            m_max = 0

            #鳥領域ノイズ除去用
            obj_count_up = 0
            obj_count_down = 0
            obj_count_left = 0
            obj_count_right = 0

            #jetcam[]の全要素にアクセス
            for i in range(len(jetcam)):
                for m in range(150):
                    #R値 >= B値 かつ R値 >= G値
                    if jetcam[i][m,2] >= jetcam[i][m,0] and jetcam[i][m,2] >= jetcam[i][m,1]:
                        #赤色にマッピングされている箇所が背景か物体か判別
                        if i <= 8:
                            for n in range(len(object_index)):
                                #物体領域ならカットしない
                                if object_index[n][0] == m and object_index[n][1] == i:
                                    obj_count_up += 1    
                                #背景領域ならカット
                                else:        
                                    i_min = 1
                                    crop_yup = ori_h / 8
                        if i >= 142:
                            for n in range(len(object_index)):
                                #物体領域ならカットしない
                                if object_index[n][0] == m and object_index[n][1] == i:
                                    obj_count_down += 1 
                                #背景領域ならカット
                                else:
                                    i_max = 1
                                    crop_ydown = ori_h - (ori_h / 8)
                        if m <= 8:
                            for n in range(len(object_index)):
                                #物体領域ならカットしない
                                if object_index[n][0] == m and object_index[n][1] == i:
                                    obj_count_left += 1 
                                #背景領域ならカット
                                else:
                                    m_min = 1
                                    crop_xleft = ori_w / 8
                        if m >= 142:
                            for n in range(len(object_index)):
                                #物体領域ならカットしない
                                if object_index[n][0] == m and object_index[n][1] == i:
                                    obj_count_right += 1 
                                #背景領域ならカット
                                else:
                                    m_max = 1
                                    crop_xright = ori_w - (ori_w / 8)
            
            #ノイズ除去
            if obj_count_up >= 10:
                i_min = 0
                crop_yup = 0
            if obj_count_down >= 10:
                i_max = 0
                crop_ydown = 0
            if obj_count_left >= 10:
                m_min = 0
                crop_xleft = 0
            if obj_count_right >= 10:
                m_max = 0
                crop_xright = 0

            if i_min == 0:
                crop_yup = 0
            if i_max == 0:
                crop_ydown = ori_h
            if m_min == 0:
                crop_xleft = 0
            if m_max == 0:
                crop_xright = ori_w

            #提案手法, 入力画像削減
            im_crop = original_im.crop((int(crop_xleft),int(crop_yup),int(crop_xright),int(crop_ydown)))
            crop_image = './crop_image/crop_image_' + root + '.jpg'
            im_crop.save(crop_image)                    
            if i_min == 0 and i_max == 0 and m_min == 0 and m_max == 0:
                print("original!")
                crop_input = load_image(abs_name)
            else:
                print("cut!")
                crop_input = load_image(crop_image)
            
            top_n = 200

            #削減後画像, classfication
            crop_pred = model.predict(crop_input)[0]
            crop_top = crop_pred.argsort()[-top_n:][::-1]
            result = [(classes[i], crop_pred[i]) for i in crop_top]
            rank_count_neo = 0
            print("cutting_result")

            for x in result:
                rank_count_neo += 1

                #分類したクラスのインデックス取得
                class_name_neo,rlt = x
                cls_index_neo = class_name_neo.split(".")

                #適合率算出
                if dir_name[0] == cls_index_neo[0]:
                    rank_score_neo += 1 / rank_count_neo
                    score_neo = 1 / rank_count_neo
                    print("\n")
                    print("rank_neo")
                    print(rank_count_neo)
                    print("\n")
                    print("score_neo")
                    print(score_neo)

            print("\n")

            cls = -1

            #cam_image save
            cam_name = './cam_image/' + root + '.jpg'
            cv2.imwrite(cam_name,np.uint8(jetcam))
            cv2.imwrite('guided_backprop.jpg', deprocess_image(gb[0]))
            cv2.imwrite('guided_gradcam.jpg', deprocess_image(guided_gradcam[0]))
        
        #visuallize
        if visualize:
            plt.figure(figsize=(15, 10))
            plt.subplot(131)
            plt.title('GradCAM')
            plt.axis('off')
            plt.imshow(load_image(abs_name, preprocess=False))
            plt.imshow(gradcam, cmap='jet', alpha=0.5)

            plt.subplot(132)
            plt.title('Guided Backprop')
            plt.axis('off')
            plt.imshow(np.flip(deprocess_image(gb[0]), -1))
            
            plt.subplot(133)
            plt.title('Guided GradCAM')
            plt.axis('off')
            plt.imshow(np.flip(deprocess_image(guided_gradcam[0]), -1))
            plt.show()
            plt.close()


    
    #total result
    print("\n")
    Compliance_rate = rank_score / 8
    print("Compliance_rate")
    print(Compliance_rate)
    print("\n")
    print("Compliance_rate_neo")
    Compliance_rate_neo = rank_score_neo / 8
    print(Compliance_rate_neo)
    print("\n")
    print("end_classfication!")
    return gradcam, gb, guided_gradcam

if __name__ == '__main__':
    model = build_model()
    guided_model = build_guided_model()
    gradcam, gb, guided_gradcam = compute_saliency(model, guided_model, layer_name='block5_conv3',
                                             img_path=sys.argv[1], cls=-1, visualize=True, save=True)
