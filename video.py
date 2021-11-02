#-*-coding:utf-8-*-
# date:2021-03-16
# Author: Eric.Lee
# function: yolo v5 video inference

import warnings
warnings.filterwarnings("ignore")
import argparse
from utils.datasets import *
from utils.utils import *
import time

def detect(save_img=False):
    # 解析配置参数
    source, weights, half, imgsz = \
        opt.source, opt.weights, opt.half, opt.img_size
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')
    # 初始化 模型推理硬件
    device = torch_utils.select_device(opt.device)

    # 模型加载初始化
    model = torch.load(weights, map_location=device)['model']
    # 模型设置为推理模式
    model.to(device).eval()
    # Cpu 半精度 flost16 推理设置 ：Half precision
    half = half and device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    # Set Dataloader
    vid_path, vid_writer = None, None

    # Get names and colors
    names = model.names if hasattr(model, 'names') else model.modules.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img.float()) if device.type != 'cpu' else None  # run once

    vid_cap = cv2.VideoCapture(source)
    save_path = "./demo/"
    while True:
        ret, img0 = vid_cap.read()
        if ret:
            # 输入图像预处理
            img = letterbox(img0, new_shape=imgsz)[0]
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)

            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # 模型推理
            t1 = torch_utils.time_synchronized()
            pred = model(img, augment=opt.augment)[0]
            t2 = torch_utils.time_synchronized()

            if half:
                pred = pred.float()

            # NMS 操作
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres,
                                       fast=True, classes=opt.classes, agnostic=opt.agnostic_nms)
            print("opt.classes",opt.classes,opt.agnostic_nms,opt.augment)
            # 输出检测结果
            for i, det in enumerate(pred):  # detections per image

                p, s, im0 = source, '', img0

                s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  #  normalization gain whwh
                if det is not None and len(det):
                    # 推理的图像分辨率转为原图分辨率：Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += '%g %ss, ' % (n, names[int(c)])  # add to string
                    # Write results
                    output_dict_ = []
                    for *xyxy, conf, cls in det:

                        x1,y1,x2,y2 = xyxy
                        output_dict_.append((float(x1),float(y1),float(x2),float(y2)))
                        label = '%s %.2f' % (names[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                    print("output_dict_ : ",output_dict_)
                print('%sDone. (%.3fs)' % (s, t2 - t1))

            cv2.putText(im0, "YoLoV5-Detect Hands", (5,im0.shape[0]-3),cv2.FONT_HERSHEY_DUPLEX, 1.2, (55, 0, 220),7)
            cv2.putText(im0, "YoLoV5-Detect Hands", (5,im0.shape[0]-3),cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 50, 50),2)

            cv2.namedWindow("video",0)
            cv2.imshow("video", im0)
            if cv2.waitKey(1) == 27:  # Esc to quit
                vid_writer.release()
                raise StopIteration

            if vid_writer is None: # 记录 demo 视频
                fps = 25
                w = int(im0.shape[1])
                h = int(im0.shape[0])
                loc_time = time.localtime()
                time_str = time.strftime("%Y-%m-%d_%H-%M-%S", loc_time)
                vid_writer = cv2.VideoWriter(save_path+"_{}.mp4".format(time_str), cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
            vid_writer.write(im0)

    vid_cap.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='model_exp_hand_x/hand_x.pt', help='model.pt path')
    parser.add_argument('--source', type=str, default="./video/1.mp4", help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.31, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--half', default=False, help='half precision FP16 inference')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', default=False, help='augmented inference')
    opt = parser.parse_args()
    print(opt) # 打印输入配置参数

    with torch.no_grad():
        detect(save_img = True)
