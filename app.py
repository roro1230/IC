from flask import Flask,request, jsonify, send_file
from PIL import Image
import torch
import torchvision.transforms as transforms
import io
import os
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import transforms
from torch import optim, nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from tqdm.notebook import tqdm
import kornia.color as color
from datetime import datetime
from pathlib import Path
import os
from pathlib import Path
from torchvision.transforms import ToPILImage
import os
from DCGAN import DCGenerator, Preprocessor
from CNN import CNNColorization
from GAN import Generator



# Khởi tạo Flask app
app = Flask(__name__)

# Khởi tạo Flask app
app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'
OUTPUT_FOLDER = 'static/outputs/'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Đảm bảo thư mục tồn tại
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def normalize_lab(L, ab):
    """
    Normalize the L and ab channels of an image in Lab color space.
    (Even though ab channels are in [-128, 127] range, divide them by 110 because higher values are very rare. 
    This makes the distribution closer to [-1, 1] in most cases.)
    """
    L = L / 50. - 1.
    ab = ab / 110.
    return L, ab

def denormalize_lab(L, ab):
    """
    Denormalize the L and ab channels of an image in Lab color space.
    (reverse of normalize_lab function)
    """
    L = (L + 1) * 50.
    ab = ab * 110.
    return L, ab

# Class below is taken from a notebook of a similar project
# https://colab.research.google.com/github/moein-shariatnia/Deep-Learning/blob/main/Image%20Colorization%20Tutorial/Image%20Colorization%20with%20U-Net%20and%20GAN%20Tutorial.ipynb

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.count, self.avg, self.sum = [0.] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += count * val
        self.avg = self.sum / self.count

# Hàm xử lý grayscale
def convert_to_grayscale(image_path):
    image = Image.open(image_path).convert('L')  # Chuyển ảnh sang grayscale
    grayscale_path = os.path.join(app.config['OUTPUT_FOLDER'], 'grayscale_' + os.path.basename(image_path))
    image.save(grayscale_path)
    return grayscale_path

class ColorizationDataset(Dataset):
    """
    A dataset class for image colorization with a single image input.
    
    Attributes:
    -----------
    image_path: str
        - The path to the image file.
    transform: torchvision.transforms.Compose
        - The transformations to apply to the image.
    
    Methods:
    --------
    __init__: 
        - Initialize the dataset with a single image.
        - Define transformations for the dataset.
        
    __len__:
        - Return 1 because we only have one image.
        
    __getitem__:
        - Load the image, apply transformations, convert to Lab color space, normalize, and return L and ab channels.
    """
    def __init__(self, image_path, split='val'):
        self.image_path = image_path  # Chỉ nhận 1 tệp ảnh
        self.split = split

        # Định nghĩa phép biến đổi cho ảnh
        if split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),  # Chuyển ảnh sang tensor
            ])
        elif split == 'val':
            self.transform = transforms.Compose([
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=Image.BICUBIC),
                transforms.ToTensor(),  # Chuyển ảnh sang tensor
            ])
        
    def __len__(self):
        return 1  # Vì chỉ có một ảnh

    def __getitem__(self, index):
        if index != 0:
            raise IndexError("Index out of range for single image dataset.")

        # Tải ảnh từ đường dẫn
        img = Image.open(self.image_path).convert('RGB')

        # Áp dụng các phép biến đổi
        img = self.transform(img)
        
        # Chuyển ảnh sang không gian Lab
        img_lab = color.rgb_to_lab(img)
        
        # Tách kênh L và ab
        L = img_lab[[0], ...]  # Kênh L
        ab = img_lab[[1, 2], ...]  # Kênh ab
        
        # Normalize các kênh L và ab
        L, ab = normalize_lab(L, ab)
        
        return L, ab



# Đường dẫn đến các mô hình

MODEL_PATHS = {
    'DCGAN': 'model/DCGAN.pth',
    'CNN': 'model/CNN.pth',
}

IMAGE_SIZE = 256


models = {}

for model_name, model_path in MODEL_PATHS.items():
    if os.path.exists(model_path) and model_name == 'DCGAN':
        model = DCGenerator()
        state_dict = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
        model.load_state_dict(state_dict)
        model.eval()
        models[model_name] = model
    elif os.path.exists(model_path) and model_name == 'CNN':
        model = torch.load('model/CNN.pth', map_location=torch.device('cpu'), weights_only=False)
        model.to(torch.device('cpu'))
        model.eval()
        models[model_name] = model
    else:
        raise FileNotFoundError(f"Model file '{model_path}' not found.")


@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Image Colorization</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background: #f4f4f9;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            margin: 0;
            overflow: hidden; 
            position: relative;
        }
        .container {
            text-align: center;
            background: #ffffff;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.2);
            max-width: 400px;
            width: 90%;
        }
        h1 {
            color: #333;
            margin-bottom: 20px;
        }
        form {
            margin-top: 20px;
        }
        input[type="file"] {
            display: block;
            margin: 15px auto;
        }
        button {
            background: #007BFF;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            transition: background 0.3s;
        }
        button:hover {
            background: #0056b3;
        }
        .header {
            background: #6a11cb;
            padding: 20px;
            text-align: center;
            border-bottom: 2px solid #f4f4f9;
            box-shadow: 0px 2px 5px rgba(0, 0, 0, 0.1);
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            z-index: 1000;
        }
        .header .funny-text {
            font-size: 18px;
            font-weight: bold;
            color: #fff;
            margin: 5px 0;
        }
        .container {
            margin-top: 150px;
            text-align: center;
        }
        .side-image {
            position: absolute;
            top: 50%;
            transform: translateY(-50%);
            object-fit: cover;
            opacity: 0.6;
            z-index: -1;
            max-width: 500px; /* Giảm kích thước của ảnh */
            max-height: 1000px;
        }
        .left-image {
            left: 20px;
        }
        .right-image {
            right: 20px;
        }
    </style>
</head>
<body>
    <img src="/static/side_left1.jpg" alt="Left Image" class="side-image left-image">
    <img src="/static/grayscale_side_left1.jpg" alt="Right Image" class="side-image right-image">
    <div class="header">
        <p class="funny-text">Bored of black and white? Let us sprinkle some color magic!</p>
        <p class="funny-text">Transform your dull images into vibrant masterpieces!</p>
    </div>
    <div class="container">
        <h1>Upload an Image</h1>
        <form action="/upload" method="post" enctype="multipart/form-data">
            <input type="file" name="image" accept="image/*" required>
            <button type="submit">Upload and Process</button>
        </form>
    </div>
</body>
</html>

    '''

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return "No image file provided.", 400

    # Lấy file ảnh
    image_file = request.files['image']
    original_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
    image_file.save(original_path)
    # Tiền xử lý ảnh
    grayscale_image = Image.open(image_file).convert('L')
    grayscale_path = os.path.join(app.config['OUTPUT_FOLDER'], 'grayscale_' + os.path.basename(image_file.filename))
    grayscale_image.save(grayscale_path)
    
    model_name = 'CNN'      
    # model_name = 'GAN'        # Sửa đổi mô hình thử nghiệm ở đây
    # model_name = 'DCGAN'
    model = models[model_name]  
    model.eval()  

    dataset = ColorizationDataset(image_file, split='val')  # Sử dụng dataset cho ảnh đầu vào
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)  # Đặt batch_size=1, shuffle=False

    # Lấy batch đầu tiên từ DataLoader
    L, ab = next(iter(data_loader))  
    
    with torch.no_grad():
        if model_name == 'CNN':
            fake_ab = model(L)  
            fake_ab = fake_ab.detach()  
            fake_ab = F.interpolate(fake_ab, size=(L.shape[2], L.shape[3]), mode='bilinear', align_corners=False)
        elif model_name == 'DCGAN':
            processer = Preprocessor()
            fake_L = processer(L)
            fake_ab = model(fake_L)
            fake_ab = fake_ab.detach()  
        else:
            fake_ab = model(L)  
            fake_ab = fake_ab.detach()  

    L, ab = denormalize_lab(L, ab)  
    _, fake_ab = denormalize_lab(0, fake_ab)  
    fake_imgs = color.lab_to_rgb(torch.cat([L, fake_ab], dim=1)).permute(0, 2, 3, 1).cpu()  

    L = L.permute(0, 2, 3, 1).cpu()  

    
    print("Fake images shape:", fake_imgs.shape)

    fake_img_single = fake_imgs[0]  
    fake_img_single = fake_img_single.permute(2, 0, 1)  

    print("Single fake image shape:", fake_img_single.shape)

    to_pil = ToPILImage()
    image_to_save = to_pil(fake_img_single)
    image_to_save.save("fake_image_output.png")  # Lưu ảnh vào file

    
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], 'colorized_' + os.path.basename(image_file.filename))
    image_to_save.save(output_path)

    return f'''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Image Results</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                background: linear-gradient(135deg, #2575fc, #28a745);
                color: #ffffff;
                text-align: center;
                padding: 20px;
            }}
            h1 {{
                font-size: 8rem;
                margin-bottom: 20px;
                text-transform: uppercase;
                letter-spacing: 2px;
            }}

            .image-container {{
                display: flex;
                flex-wrap: wrap;
                justify-content: center;
                gap: 20px;
            }}
            .image-card {{
                background: rgba(255, 255, 255, 0.1);
                backdrop-filter: blur(10px);
                padding: 15px;
                border-radius: 15px;
                box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.3);
                text-align: center;
                max-width: 300px;
                transition: transform 0.3s, box-shadow 0.3s;
            }}
            .image-card:hover {{
                transform: translateY(-10px);
                box-shadow: 0px 8px 25px rgba(0, 0, 0, 0.5);
            }}

            img {{
                max-width: 100%;
                border-radius: 10px;
                border: 2px solid rgba(255, 255, 255, 0.5);
                transition: transform 0.3s;
            }}
            img:hover {{
                transform: scale(1.05);
            }}

            a {{
                display: inline-block;
                margin-top: 20px;
                padding: 12px 25px;
                background: #ff6f61;
                color: #ffffff;
                text-decoration: none;
                border-radius: 25px;
                font-weight: bold;
                box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.2);
                transition: background 0.3s, box-shadow 0.3s;
            }}
            a:hover {{
                background: #ff4a3c;
                box-shadow: 0px 8px 20px rgba(0, 0, 0, 0.4);
            }}
        </style>
    </head>
    <body>
        <h1>Results</h1>
        <div class="image-container">
            <div class="image-card">
                <h3>Original Image</h3>
                <img src="/static/uploads/{os.path.basename(original_path)}" alt="Original Image">
            </div>
            <div class="image-card">
                <h3>Grayscale Image</h3>
                <img src="/static/outputs/{os.path.basename(grayscale_path)}" alt="Grayscale Image">
            </div>
            <div class="image-card">
                <h3>Colorized Image</h3>
                <img src="/static/outputs/{os.path.basename(output_path)}" alt="Colorized Image">
            </div>
        </div>
        <a href="/">Upload Another Image</a>
    </body>
    </html>
    '''


if __name__ == '__main__':
    app.run(debug=True)
