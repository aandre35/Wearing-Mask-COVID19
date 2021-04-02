import numpy as np
import PIL
from PIL import Image
import os, sys
from keras.utils import np_utils

def resize_and_save_to_numpy(IMAGE_SIZE, BOX_PER_W_IMAGE, BOX, NUM_CLASSES):
  ds_path = "./data/"
  images_paths = [ds_path+"train/images/", ds_path+"test/images/", ds_path+"valid/images/"]
  labels_paths = [ds_path+"train/labels/", ds_path+"test/labels/", ds_path+"valid/labels/"]
  x_train = np.zeros((106, IMAGE_SIZE, IMAGE_SIZE, 3))
  x_val = np.zeros((16+28, IMAGE_SIZE, IMAGE_SIZE, 3))
  y_train = np.zeros((106, BOX_PER_W_IMAGE,BOX_PER_W_IMAGE,BOX, 1+4*1+NUM_CLASSES))
  y_val = np.zeros((16+28, BOX_PER_W_IMAGE,BOX_PER_W_IMAGE,BOX, 1+4*1+NUM_CLASSES))

  for k in range(3):
    images_path=images_paths[k]
    images_dirs = os.listdir(images_path)
    images_dirs.sort()
    if k==2:
      i=15
    else:
      i=0
    for item in images_dirs:
      extension =item.split(".")[-1]
      if extension=="jpg" or extension=="JPG":
        img = Image.open(images_path+item)
        #print(img.size)
        f, e = os.path.splitext(images_path+item)
        img = img.resize((IMAGE_SIZE,IMAGE_SIZE), Image.ANTIALIAS)
        #print(img.size)
        if k==0:
          x_train[i]=np.asarray(img)
        else:
          x_val[i]=np.asarray(img)
        i+=1
      else:
        print("extension trouvée : ", extension)
    
    labels_path=labels_paths[k]
    labels_dirs = os.listdir(labels_path)
    labels_dirs.sort()
    if k==2:
      i=15
    else:
      i=0
    for item in labels_dirs:
      #print("IMAGE NUM ", i)
      extension =item.split(".")[-1]
      if extension=="txt":
        labels = open(labels_path+item, "r")
        labels= labels.read().split('\n')
        for label in labels:
            label=label.split()
            #print(label)
            ax, ay= float(label[1])*BOX_PER_W_IMAGE, float(label[2])*BOX_PER_W_IMAGE
            ind_x, ind_y = int(ax), int(ay)
            
            pred=np.array([1], dtype="i")
            classes = np_utils.to_categorical(label[0], num_classes=2)
            #classes = np.array(label[0], dtype="i")
            label[1] = ax - ind_x
            label[2] = ay - ind_y
            coordonnees = np.array(label[1:],dtype="f")
            arr_labels=np.append(np.append(pred,coordonnees),classes)
            #print(arr_labels)
            if k==0:
              p=0
              while p<BOX:
                if (y_train[i, ind_x, ind_y, p, 0] == 0 or p==BOX-1):
                  y_train[i, ind_x, ind_y, p]= arr_labels
                  break;
                else:
                  p+=1
            else:
              p=0
              while p<BOX:
                if (y_val[i, ind_x, ind_y, p, 0] == 0 or p==BOX-1):
                  y_val[i, ind_x, ind_y, p]= arr_labels
                  break;
                else:
                  p+=1
        i+=1
      else:
        print("extension trouvée : ", extension)
  return x_train/256,x_val/256,y_train, y_val
