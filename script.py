import time
from model.yolov4.tf import YOLOv4

yolo = YOLOv4(tiny=True)
yolo.classes = "coco.names"
yolo.make_model()
yolo.load_weights("./yolov4-tiny/yolov4-tiny.weights", weights_type="yolo")

start_time = time.time()
# everything is happening in this function
yolo.inference(media_path='../../../../../Affan/Downloads/boruto.mkv', is_image=False)
end_time = time.time()


def timer(start, end):
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Total time taken ")
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))


timer(start_time, end_time)
