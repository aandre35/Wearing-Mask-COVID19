import tensorflow as tf


def yolo_loss(y_true, y_pred, GRID_H, GRID_W, BOX, NO_OBJECT_SCALE, OBJECT_SCALE, OBJ_THRESHOLD, COORD_SCALE, CLASS_WEIGHTS, WARM_UP_BATCHES, CLASS_SCALE, true_boxes):
  mask_shape = tf.shape(y_true)[:4]

  cell_x = tf.cast(tf.reshape(tf.tile(tf.range(GRID_W), [GRID_H]), (1, GRID_H, GRID_W, 1, 1)), dtype=tf.float32)
  cell_y = tf.transpose(cell_x, (0,2,1,3,4))

  cell_grid = tf.tile(tf.concat([cell_x,cell_y], -1), [tf.shape(y_true)[0], 1, 1, BOX, 1])
  
  coord_mask = tf.zeros(mask_shape)
  conf_mask  = tf.zeros(mask_shape)
  class_mask = tf.zeros(mask_shape)
  
  seen = tf.Variable(0.)
  
  total_AP = tf.Variable(0.)
  
  """
  Adjust prediction
  """
  ### adjust x and y      
  pred_box_xy = tf.sigmoid(y_pred[..., 1:3]) + cell_grid
  
  ### adjust w and h
  pred_box_wh = tf.exp(y_pred[..., 3:5]) #* np.reshape(ANCHORS, [1,1,1,BOX,2])
  
  ### adjust confidence
  pred_box_conf = tf.sigmoid(y_pred[..., 0])
  
  ### adjust class probabilities
  pred_box_class = y_pred[..., 5:]
  
  """
  Adjust ground truth
  """
  ### adjust x and y
  true_box_xy = y_true[..., 1:3] # relative position to the containing cell
  
  ### adjust w and h
  true_box_wh = y_true[..., 3:5] # number of cells accross, horizontally and vertically
  
  ### adjust confidence
  true_wh_half = true_box_wh / 2.
  true_mins    = true_box_xy - true_wh_half
  true_maxes   = true_box_xy + true_wh_half
  
  pred_wh_half = pred_box_wh / 2.
  pred_mins    = pred_box_xy - pred_wh_half
  pred_maxes   = pred_box_xy + pred_wh_half       
  
  intersect_mins  = tf.maximum(pred_mins,  true_mins)
  intersect_maxes = tf.minimum(pred_maxes, true_maxes)
  intersect_wh    = tf.maximum(intersect_maxes - intersect_mins, 0.)
  intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]
  
  true_areas = true_box_wh[..., 0] * true_box_wh[..., 1]
  pred_areas = pred_box_wh[..., 0] * pred_box_wh[..., 1]

  union_areas = pred_areas + true_areas - intersect_areas
  iou_scores  = tf.truediv(intersect_areas, union_areas)
  
  true_box_conf = iou_scores * y_true[..., 0]
  
  ### adjust class probabilities
  true_box_class = tf.argmax(y_true[...,5:], -1)
  true_box_class = tf.cast(true_box_class, dtype=tf.int32)    
  """
  Determine the masks
  """
  ### coordinate mask: simply the position of the ground truth boxes (the predictors)
  coord_mask = tf.expand_dims(y_true[..., 0], axis=-1) * COORD_SCALE
  
  ### confidence mask: penelize predictors + penalize boxes with low IOU
  # penalize the confidence of the boxes, which have IOU with some ground truth box < 0.6
  true_xy = true_boxes[..., 0:2]
  true_wh = true_boxes[..., 2:4]
  
  true_wh_half = true_wh / 2.
  true_mins    = true_xy - true_wh_half
  true_maxes   = true_xy + true_wh_half
  
  pred_xy = tf.expand_dims(pred_box_xy, 4)
  pred_wh = tf.expand_dims(pred_box_wh, 4)
  
  pred_wh_half = pred_wh / 2.
  pred_mins    = pred_xy - pred_wh_half
  pred_maxes   = pred_xy + pred_wh_half    
  
  intersect_mins  = tf.maximum(pred_mins,  true_mins)
  intersect_maxes = tf.minimum(pred_maxes, true_maxes)
  intersect_wh    = tf.maximum(intersect_maxes - intersect_mins, 0.)
  intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]
  
  true_areas = true_wh[..., 0] * true_wh[..., 1]
  pred_areas = pred_wh[..., 0] * pred_wh[..., 1]

  union_areas = pred_areas + true_areas - intersect_areas
  iou_scores  = tf.truediv(intersect_areas, union_areas)

  best_ious = tf.reduce_max(iou_scores, axis=4)
  conf_mask = conf_mask + tf.cast(best_ious < 0.6,dtype=tf.float32) * (1 - y_true[..., 0]) * NO_OBJECT_SCALE
  
  # penalize the confidence of the boxes, which are reponsible for corresponding ground truth box
  conf_mask = conf_mask + y_true[..., 0] * OBJECT_SCALE
  
  ### class mask: simply the position of the ground truth boxes (the predictors)
  class_mask = y_true[..., 0] * tf.gather(CLASS_WEIGHTS, true_box_class) * CLASS_SCALE       
  
  """
  Warm-up training
  """
  no_boxes_mask = tf.cast(coord_mask < COORD_SCALE/2., dtype=tf.float32)
  seen = tf.assign_add(seen, 1.)
  
  true_box_xy, true_box_wh, coord_mask = tf.cond(tf.less(seen, WARM_UP_BATCHES), 
                        lambda: [true_box_xy + (0.5 + cell_grid) * no_boxes_mask, 
                                  true_box_wh + tf.ones_like(true_box_wh) * \
                                  #np.reshape(ANCHORS, [1,1,1,BOX,2]) * \
                                  no_boxes_mask, 
                                  tf.ones_like(coord_mask)],
                        lambda: [true_box_xy, 
                                  true_box_wh,
                                  coord_mask])
  
  """
  Finalize the loss
  """
  nb_coord_box = tf.reduce_sum(tf.cast(coord_mask > 0.0,dtype=tf.float32))
  nb_conf_box  = tf.reduce_sum(tf.cast(conf_mask  > 0.0,dtype=tf.float32))
  nb_class_box = tf.reduce_sum(tf.cast(class_mask > 0.0,dtype=tf.float32))
  

  loss_xy    = tf.reduce_sum(tf.square(true_box_xy-pred_box_xy)     * coord_mask) / (nb_coord_box + 1e-6) / 2.
  
  loss_wh    = tf.reduce_sum(tf.square(true_box_wh-pred_box_wh)     * coord_mask) / (nb_coord_box + 1e-6) / 2.
  loss_conf  = tf.reduce_sum(tf.square(true_box_conf-pred_box_conf) * conf_mask)  / (nb_conf_box  + 1e-6) / 2.
  
  loss_class = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_box_class, logits=pred_box_class)
  loss_class = tf.reduce_sum(loss_class * class_mask) / (nb_class_box + 1e-6)
  
  loss = loss_xy + loss_wh + loss_conf + loss_class

  nb_true_box = tf.reduce_sum(y_true[..., 0])
  nb_pred_box = tf.reduce_sum(tf.cast(true_box_conf > 0.5, dtype=tf.float32) * tf.cast(pred_box_conf > OBJ_THRESHOLD,dtype=tf.float32))
  
  total_AP = tf.assign_add(total_AP, nb_pred_box/nb_true_box) 
  
  loss = tf.Print(loss, [loss_xy, loss_wh, loss_conf, loss_class, loss, total_AP/seen], message='DEBUG', summarize=1000)
  
  return loss