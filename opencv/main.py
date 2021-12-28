import cv2
import numpy as np
import argparse
import math

class my_nanodet():
    def __init__(self, modelpath, classfile, input_shape=320, prob_threshold=0.4, iou_threshold=0.3):
        self.classes = list(map(lambda x: x.strip(), open(classfile).readlines()))
        self.num_classes = len(self.classes)
        self.strides = (8, 16, 32, 64)
        self.input_shape = (input_shape, input_shape)
        self.reg_max = 7
        self.prob_threshold = prob_threshold
        self.iou_threshold = iou_threshold
        self.project = np.arange(self.reg_max + 1)
        self.mean = np.array([103.53, 116.28, 123.675], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([57.375, 57.12, 58.395], dtype=np.float32).reshape(1, 1, 3)
        self.net = cv2.dnn.readNet(modelpath)
        self.keep_ratio = False
        self.mlvl_anchors = []
        for i in range(len(self.strides)):
            anchors = self._make_grid(
                (math.ceil(self.input_shape[0] / self.strides[i]), math.ceil(self.input_shape[1] / self.strides[i])),
                self.strides[i])
            self.mlvl_anchors.append(anchors)
    
    def _make_grid(self, featmap_size, stride):
        feat_h, feat_w = featmap_size
        shift_x = np.arange(0, feat_w) * stride
        shift_y = np.arange(0, feat_h) * stride
        xv, yv = np.meshgrid(shift_x, shift_y)
        xv = xv.flatten()
        yv = yv.flatten()
        return np.stack((xv, yv), axis=-1)
        # cx = xv + 0.5 * (stride - 1)
        # cy = yv + 0.5 * (stride - 1)
        # return np.stack((cx, cy), axis=-1)
    
    def softmax(self, x, axis=1):
        x_exp = np.exp(x)
        # 如果是列向量，则axis=0
        x_sum = np.sum(x_exp, axis=axis, keepdims=True)
        s = x_exp / x_sum
        return s
    
    def _normalize(self, img):
        img = img.astype(np.float32)
        img = (img - self.mean) / self.std
        return img
    
    def resize_image(self, srcimg, keep_ratio=True):
        top, left, newh, neww = 0, 0, self.input_shape[0], self.input_shape[1]
        if keep_ratio and srcimg.shape[0] != srcimg.shape[1]:
            hw_scale = srcimg.shape[0] / srcimg.shape[1]
            if hw_scale > 1:
                newh, neww = self.input_shape[0], int(self.input_shape[1] / hw_scale)
                img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
                left = int((self.input_shape[1] - neww) * 0.5)
                img = cv2.copyMakeBorder(img, 0, 0, left, self.input_shape[1] - neww - left, cv2.BORDER_CONSTANT,
                                         value=0)  # add border
            else:
                newh, neww = int(self.input_shape[0] * hw_scale), self.input_shape[1]
                img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
                top = int((self.input_shape[0] - newh) * 0.5)
                img = cv2.copyMakeBorder(img, top, self.input_shape[0] - newh - top, 0, 0, cv2.BORDER_CONSTANT, value=0)
        else:
            img = cv2.resize(srcimg, self.input_shape, interpolation=cv2.INTER_AREA)
        return img, newh, neww, top, left
    
    def detect(self, srcimg):
        img, newh, neww, top, left = self.resize_image(srcimg, keep_ratio=self.keep_ratio)
        img = self._normalize(img)
        blob = cv2.dnn.blobFromImage(img)
        # Sets the input to the network
        self.net.setInput(blob)
        
        # Runs the forward pass to get output of the output layers
        outs = self.net.forward(self.net.getUnconnectedOutLayersNames())[0].squeeze(axis=0)
        det_bboxes, det_conf, det_classid = self.post_process(outs)
        
        drawimg = srcimg.copy()
        ratioh, ratiow = srcimg.shape[0] / newh, srcimg.shape[1] / neww
        for i in range(det_bboxes.shape[0]):
            xmin, ymin, xmax, ymax = max(int((det_bboxes[i, 0] - left) * ratiow), 0), max(
                int((det_bboxes[i, 1] - top) * ratioh), 0), min(
                int((det_bboxes[i, 2] - left) * ratiow), srcimg.shape[1]), min(int((det_bboxes[i, 3] - top) * ratioh),
                                                                               srcimg.shape[0])
            self.drawPred(drawimg, det_classid[i], det_conf[i], xmin, ymin, xmax, ymax)
        return drawimg
    
    def post_process(self, preds, scale_factor=1, rescale=False):
        mlvl_bboxes = []
        mlvl_scores = []
        ind = 0
        for stride, anchors in zip(self.strides, self.mlvl_anchors):
            cls_score, bbox_pred = preds[ind:(ind + anchors.shape[0]), :self.num_classes], preds[ind:(ind + anchors.shape[0]), self.num_classes:]
            ind += anchors.shape[0]
            bbox_pred = self.softmax(bbox_pred.reshape(-1, self.reg_max + 1), axis=1)
            # bbox_pred = np.sum(bbox_pred * np.expand_dims(self.project, axis=0), axis=1).reshape((-1, 4))
            bbox_pred = np.dot(bbox_pred, self.project).reshape(-1, 4)
            bbox_pred *= stride
            
            # nms_pre = cfg.get('nms_pre', -1)
            nms_pre = 1000
            if nms_pre > 0 and cls_score.shape[0] > nms_pre:
                max_scores = cls_score.max(axis=1)
                topk_inds = max_scores.argsort()[::-1][0:nms_pre]
                anchors = anchors[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                cls_score = cls_score[topk_inds, :]
            
            bboxes = self.distance2bbox(anchors, bbox_pred, max_shape=self.input_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(cls_score)
        
        mlvl_bboxes = np.concatenate(mlvl_bboxes, axis=0)
        if rescale:
            mlvl_bboxes /= scale_factor
        mlvl_scores = np.concatenate(mlvl_scores, axis=0)
        
        bboxes_wh = mlvl_bboxes.copy()
        bboxes_wh[:, 2:4] = bboxes_wh[:, 2:4] - bboxes_wh[:, 0:2]  ####xywh
        classIds = np.argmax(mlvl_scores, axis=1)
        confidences = np.max(mlvl_scores, axis=1)  ####max_class_confidence
        
        indices = cv2.dnn.NMSBoxes(bboxes_wh.tolist(), confidences.tolist(), self.prob_threshold, self.iou_threshold)
        if len(indices) > 0:
            mlvl_bboxes = mlvl_bboxes[indices[:, 0]]
            confidences = confidences[indices[:, 0]]
            classIds = classIds[indices[:, 0]]
            return mlvl_bboxes, confidences, classIds
        else:
            print('nothing detect')
            return np.array([]), np.array([]), np.array([])
    
    def distance2bbox(self, points, distance, max_shape=None):
        x1 = points[:, 0] - distance[:, 0]
        y1 = points[:, 1] - distance[:, 1]
        x2 = points[:, 0] + distance[:, 2]
        y2 = points[:, 1] + distance[:, 3]
        if max_shape is not None:
            x1 = np.clip(x1, 0, max_shape[1])
            y1 = np.clip(y1, 0, max_shape[0])
            x2 = np.clip(x2, 0, max_shape[1])
            y2 = np.clip(y2, 0, max_shape[0])
        return np.stack([x1, y1, x2, y2], axis=-1)
    
    def drawPred(self, frame, classId, conf, left, top, right, bottom):
        # Draw a bounding box.
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), thickness=4)
        
        label = '%.2f' % conf
        label = '%s:%s' % (self.classes[classId], label)
        
        # Display the label at the top of the bounding box
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        # cv.rectangle(frame, (left, top - round(1.5 * labelSize[1])), (left + round(1.5 * labelSize[0]), top + baseLine), (255,255,255), cv.FILLED)
        cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2)
        return frame


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgpath', type=str, default='imgs/person.jpg', help="image path")
    parser.add_argument('--input_shape', default=320, type=int, choices=[320, 416], help='input image shape')
    parser.add_argument('--modelpath', type=str, default='onnxmodel/nanodet-plus-m_320.onnx',
                        choices=["onnxmodel/nanodet-plus-m_320.onnx", "onnxmodel/nanodet-plus-m_416.onnx",
                                 "onnxmodel/nanodet-plus-m-1.5x_320.onnx", "onnxmodel/nanodet-plus-m-1.5x_416.onnx"],
                        help="onnx filepath")
    parser.add_argument('--classfile', type=str, default='onnxmodel/coco.names', help="classname filepath")
    parser.add_argument('--confThreshold', default=0.4, type=float, help='class confidence')
    parser.add_argument('--nmsThreshold', default=0.6, type=float, help='nms iou thresh')
    args = parser.parse_args()
    
    srcimg = cv2.imread(args.imgpath)
    net = my_nanodet(args.modelpath, args.classfile, input_shape=args.input_shape, prob_threshold=args.confThreshold, iou_threshold=args.nmsThreshold)
    srcimg = net.detect(srcimg)
    
    winName = 'Deep learning object detection in OpenCV'
    cv2.namedWindow(winName, cv2.WINDOW_NORMAL)
    cv2.imshow(winName, srcimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
