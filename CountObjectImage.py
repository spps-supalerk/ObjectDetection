import numpy as np
import cv2
import matplotlib.pyplot as plt

class CountObjectImage():

    def __init__(self, net, labels, confident_constant = 0.5, threshold_constant = 0.3, img_width = 416, img_hight = 416):
        
        self.labels = labels
        self.net = net
        self.confident_constant = confident_constant
        self.threshold_constant = threshold_constant
        self.img_width = img_width
        self.img_hight = img_hight
        self.img = None
        self.boxes = None
        self.confidences = None
        self.classIDs = None
        self.colors = np.random.randint(0, 255, size=(len(self.labels), 3), dtype=np.uint8)
    
    def fit(self, img):

        # get image
        self.img = img

        # Determine only the output layer names that we need from YOLO
        olayer_name = self.net.getLayerNames()
        olayer_name = [ olayer_name[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

        # Be careful, each model requires different preprocessing!!!
        h,w = self.img.shape[:2]
        blob = cv2.dnn.blobFromImage(self.img,
                                    1 / 255.0,                          # scaleFactor
                                    (self.img_width, self.img_hight),   # spatial size of the CNN
                                    swapRB=True, crop=False)
        # Pass the blob to the network
        self.net.setInput(blob)
        outputs = self.net.forward(olayer_name)

        # Lists to store detected bounding boxes, confidences and classIDs
        self.boxes = []
        self.confidences = []
        self.classIDs = []

        # Loop over each of the layer outputs
        for output in outputs:
            # Loop over each of the detections
            for detection in output:
                # Extract the confidence (i.e., probability) and classID
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
            
                # Filter out weak detections by ensuring the confidence is greater than the threshold
                if confidence < self.confident_constant:
                    continue

                # Compute the (x, y)-coordinates of the bounding box
                box = detection[0:4] * np.array( [w,h,w,h] )
                (centerX, centerY, width, height) = box.astype('int')
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # Add a new bounding box to our list
                self.boxes.append([x, y, int(width), int(height)])
                self.confidences.append(float(confidence))
                self.classIDs.append(classID)

    def count_obj(self):

        # Count Detected Object
        labels_map = [self.labels[i] for i in self.classIDs]

        # Count each labels
        object_count = {}  
        for l in labels_map:
            if l in object_count:
                object_count[l] += 1
            else:
                object_count[l] = 1

        # Return dictionary of obj count
        return object_count

    def image_show(self, figsize=(12,8), title = 'Image After Doing Object Detection\n', fontsize=25):

        # Non-maxima suppression to suppress weak, overlapping bounding boxes
        idxs = cv2.dnn.NMSBoxes(self.boxes, self.confidences, self.confident_constant, self.threshold_constant)

        # If at least one detection exists, draw the detection result(s) on the input image
        if len(idxs) > 0:
            for i in idxs.flatten():
                # Extract the bounding box coordinates
                (x, y) = (self.boxes[i][0], self.boxes[i][1])
                (w, h) = (self.boxes[i][2], self.boxes[i][3])

                # Draw a bounding box rectangle and label on the image
                color = [int(c) for c in self.colors[self.classIDs[i]]]
                cv2.rectangle(self.img, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(self.labels[self.classIDs[i]], self.confidences[i])
                cv2.putText(self.img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Display image after detected
        plt.figure(figsize = figsize)
        plt.title(title, fontsize=fontsize)
        plt.imshow( cv2.cvtColor( self.img, cv2.COLOR_BGR2RGB) )
        plt.show()

if __name__ == "__main__":
    # load image
    # print(cv2.__version__)
    dir_img = r'.\image_test.jpg'
    img = cv2.imread(dir_img)
    # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # plt.show()  

    # Load labels of model
    labels = open("./YOLO-COCO/coco.names").read().strip().split("\n")

    # Load a pre-trained YOLOv3 model from disk
    net = cv2.dnn.readNetFromDarknet("./YOLO-COCO/yolov3.cfg","./YOLO-COCO/yolov3.weights")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU) 

    # Import class CountObjectImg
    co = CountObjectImage(net = net, labels = labels)
    co.fit(img)
    print(co.count_obj())
    co.image_show()
    