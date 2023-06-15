# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

import tkinter as tk
import tkinter.ttk as exTK
from tkinter import font
from utils.torch_utils import select_device, smart_inference_mode
from utils.plots import Annotator, colors, save_one_box
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from models.common import DetectMultiBackend
import argparse
import os
import platform
import sys
from pathlib import Path
import telegram
import asyncio
import torch
import firebase_admin
from firebase_admin import credentials, firestore
from PIL import Image, ImageTk
import time
import threading

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

# Firebase

# Use a service account.
cred = credentials.Certificate("nckh2023-6e660-firebase-adminsdk-pzchp-ba92ad8282.json")

# Application Default credentials are automatically created.
app = firebase_admin.initialize_app(cred)
db = firestore.client()

col_ref = db.collection(u'components')

is_borrow = False
app = tk.Tk()
info = {}

def update_amount(label, type, amount):
    is_increase = type == 'increase'

    parent_label = label.split("/", 1)[0]

    parent_doc = col_ref.where('label', '==', parent_label).get()
    if not parent_doc:
        # Parent document not found, handle error here
        return

    child_label = label.split("/", 1)[1]

    print("CHILD_LABEL =>", child_label)

    parent_id = parent_doc[0].id
    child_doc = col_ref.document(parent_id).collection(
        'items').where('label', '==', child_label).get()

    if not child_doc:
        # Child document not found, handle error here
        return

    child_id = child_doc[0].id
    child_ref = col_ref.document(parent_id).collection(
        'items').document(child_id)
    
    # Get the current amount of the child_ref
    current_amount = child_doc[0].get("amount")

    # Check if the current amount is already 0
    if current_amount == 0 and not is_increase:
        print("Amount is already 0. No further subtraction.")
        return

    # Calculate the new amount after subtraction
    new_amount = int(current_amount) - amount if not is_increase else int(current_amount) + amount

    # Ensure the new amount is not less than 0
    new_amount = max(new_amount, 0)

    child_ref.update({'amount': new_amount})
    
    updated_child_data = child_ref.get().to_dict()
    print("Updated Child Data:", updated_child_data)


async def sen_telegram(ten_nhom, nhom_truong, msv, lop_hp,tinh_trang, amount):
    global is_borrow
    text = ''
    bot = telegram.Bot(token="6211404922:AAEBn2rI4mm92avEXpoao_xPUZpsK6NMHVg")
    chat_id = "5243841729"
    if (is_borrow):
      text = f"Tên nhóm: {ten_nhom}\nTên nhóm trưởng: {nhom_truong}\nMã sinh viên: {msv}\nLớp học phần: {lop_hp}\nSố lượng linh kiện mượn: {amount}\nTình tạng linh kiện: {tinh_trang}"
    else:
      text = f"Tên nhóm: {ten_nhom}\nTên nhóm trưởng: {nhom_truong}\nMã sinh viên: {msv}\nLớp học phần: {lop_hp}\nSố lượng linh kiện trả: {amount}\nTình tạng linh kiện: {tinh_trang}"
    await bot.send_message(chat_id=chat_id, text=text)


def nguoi_dung():
    ten_nhom = input("Mời nhập tên nhóm: ")
    nhom_truong = input("Mời nhập tên nhóm trưởng: ")
    msv = int(input("Mời nhập mã sinh viên: "))
    lop_hp = input("Mời nhập tên lớp học phần: ")
    return ten_nhom, nhom_truong, msv, lop_hp


count = 0

isConfirm = False

TIME_TO_END = 30


def checkToSendMessage():
    global count

    count += 1

    if (count == TIME_TO_END):
        count = 0


@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    global count
    global is_borrow
    global app
    

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)
        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                alo = " "
                n = 0

                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    alo += f"{n} {names[int(c)]}, "

                if count == TIME_TO_END - 1:
                    print("Đã quét xong")
                    # ten_nhom, nhom_truong, msv, lop_hp = nguoi_dung()
                    # alo += f" {n} {names[int(c)],}"
                    confirm = tk.Tk()

                    def confirm_borrow():
                        print('confirm_borrow')
                        _str = alo.split(',')[:-1]
                        global info 
                        # Split each element by space and create a list of tuples
                        lst = [(s.split()[1], int(s.split()[0])) for s in _str]
                        type = 'decrease' if is_borrow else 'increase' 
                        print("LIST =>", lst)
                        print("TYPE =>", type)
                        for item in lst:
                            label = item[0] + "/" + item[0]
                            update_amount(label=label, type=type, amount=int(item[1]))
                        
                        asyncio.run(sen_telegram(info["ten_nhom"], info["nhom_truong"], info["msv"], info["lop_hp"], info["tinh_trang"], alo))
                        confirm.destroy()
                        app.destroy()
                        
                        raise StopIteration
                        # loop = asyncio.get_event_loop()
                        # loop.run_until_complete(asyncio.wait([sen_telegram(info["ten_nhom"], info["nhom_truong"], info["msv"], info["lop_hp"], alo)]))
                    def reject_borrow():
                        confirm.destroy()
                        app.destroy()
                        raise StopIteration

                    confirm.title("Bảng xác nhận")
                    confirm.geometry("200x100")

                    if (is_borrow):
                        flag = "mượn"
                    else:
                        flag = "trả"

                    tk.Label(confirm, text="Bạn có xác nhận "+flag).pack()
                    yes_button = tk.Button(confirm, text="Có", command=confirm_borrow)
                    yes_button.place(height=30, width=90, x= 5, y= 50)
                    no_button = tk.Button(confirm, text="Không", command=reject_borrow)
                    no_button.place(height=30, width=90, x= 105, y= 50)

                    confirm.mainloop()

                    raise StopIteration

                    # app.destroy()
                    # raise StopIteration

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    # if save_txt:  # Write to file
                    #     xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    #     line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                    #     with open(f'{txt_path}.txt', 'a') as f:
                    #         f.write(('%g ' * len(line)).rstrip() % line + '\n')
                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
                #
                checkToSendMessage()

            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str("Object_Detection"), im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt

def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))

    def move_focus(event):
        if event.keysym == "Up":
            current_entry = app.focus_get()
            current_entry.tk_focusPrev().focus()
            return "break"
        elif event.keysym == "Down":
            current_entry = app.focus_get()
            current_entry.tk_focusNext().focus()
            return "break"
        elif event.keysym == "Left":
            current_entry = app.focus_get()
            next_entry = current_entry.tk_focusPrev()
            if next_entry:
                next_entry.focus_set()
            return "break"
        elif event.keysym == "Right":
            current_entry = app.focus_get()
            next_entry = current_entry.tk_focusNext()
            if next_entry:
                next_entry.focus_set()
            return "break"

    # def focus_next_entry(event):
    #     event.widget.tk_focusNext().focus()
    #     return "break"
    
    global app
    app.title("Hệ thống quản lý linh kiện")
    icon_image=Image.open('icon.png')
    icon = ImageTk.PhotoImage(icon_image)
    app.iconphoto(True,icon)
    app.resizable(False, False)
    app.geometry("650x330")
    app.bind("<Up>", move_focus)
    app.bind("<Down>", move_focus)
    app.bind("<Left>", move_focus)
    app.bind("<Right>", move_focus)

    image_path = "logo.png"
    image = Image.open(image_path)
    image = image.resize((650,50))
    photo = ImageTk.PhotoImage(image)
    logo_truong = tk.Label(app, image=photo)
    logo_truong.place(height=50,width=650,x=0,y=0)

    bold_font = font.Font(weight="bold", size=11, family="Arial")
    l_de_tai = tk.Label(app, text="ĐỀ TÀI NGUYÊN CỨU KHOA HỌC",bg="#54DB90", fg="black",font=bold_font)
    l_de_tai.pack()
    l_ten_de_tai = tk.Label(app, text="XÂY DỰNG PHẦN MỀM QUẢN LÝ LINH KIỆN ĐIỆN TỬ SỬ DỤNG XỬ LÝ ẢNH",bg="#54DB90", fg="black",font=bold_font)
    l_ten_de_tai.pack()

    l_ten_nhom = exTK.Label(app, text="Tên nhóm: ")
    ten_nhom = tk.Entry(app)
    l_ten_nhom.place(height=25,width=100,x=20,y=70)
    ten_nhom.place(height=25,width=150,x=140,y=70)
    ten_nhom.focus_set()
    # ten_nhom.bind("<Return>", focus_next_entry)

    l_nhom_truong = exTK.Label(app, text="Tên nhóm trưởng: ")
    nhom_truong = tk.Entry(app)
    l_nhom_truong.place(height=25,width=100,x=350,y=70)
    nhom_truong.place(height=25,width=150,x=470,y=70)
    # nhom_truong.bind("<Return>", focus_next_entry)

    l_msv = exTK.Label(app, text="Mã sinh viên: ")
    msv = tk.Entry(app)
    l_msv.place(height=25,width=100,x=20,y=70+65)
    msv.place(height=25,width=150,x=140,y=70+65)
    # msv.bind("<Return>", focus_next_entry)

    l_lop_hp = exTK.Label(app, text="Lớp học phần: ")
    lop_hp = tk.Entry(app)
    l_lop_hp.place(height=25,width=100,x=350,y=70+65)
    lop_hp.place(height=25,width=150,x=470,y=70+65)

    l_options = exTK.Label(app, text="Lựa chọn chế độ:")
    options = ["Mượn", "Trả"]  # Thay đổi các lựa chọn theo ý muốn
    selected_option = tk.StringVar(app)
    selected_option.set(options[0])  # Giá trị mặc định cho dropdown
    l_options.place(height=25,width=100,x=20,y=70+65+65)

    l_tinh_trang = exTK.Label(app, text = "Tình trạng linh kiện:")
    tinh_trang = tk.Entry(app)
    l_tinh_trang.place(height=25,width=120,x=350,y=70+65+65)
    tinh_trang.place(height=25,width=150,x=470,y=70+65+65)

    def run_with_params():
      global is_borrow

      if (selected_option.get() == "Mượn"):
        is_borrow = True
      else:
        is_borrow = False

      global info
      info = {
        "ten_nhom": ten_nhom.get(),
        "nhom_truong": nhom_truong.get(),
        "msv": msv.get(),
        "lop_hp": lop_hp.get(),
        "tinh_trang": tinh_trang.get()
      }
      run(**vars(opt))
      

    dropdown = tk.OptionMenu(app, selected_option, *options)
    dropdown.place(height=25,width=100,x=160,y=70+65+65)

    save_button = tk.Button(app, text="Xác nhận", command=run_with_params)
    save_button.place(height=25,width=100,x=20,y=70+65+65+65)

    quit_button = tk.Button(app, text="Exit", command=app.quit)
    quit_button.place(height=25,width=100,x=520,y=70+65+65+65)

    app.mainloop()

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
