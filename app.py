from flask import Flask, render_template, request, session
from joblib import load
import numpy as np
import tempfile
import random
import os
import cv2
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
import base64

app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

@app.route("/")
def index():
    return render_template('index.html')

@app.route('/vd')
def vd():
    return render_template('vd.html')

@app.route('/expvd')
def expvd():
    return render_template('evd.html')

@app.route('/evdentry')
def evdentry():
    return render_template('evd-entry.html')

@app.route('/vdentry')
def vdentry():
    return render_template('vd-entry.html')

@app.route('/cbentry')
def cbentry():
    return render_template('cb-entry.html')

def simulate_color_blindness(image_array, type):
    if type == 'option-1':
        image_array = simulate_deuteranomaly(image_array)
    elif type == 'option-2':
        image_array = simulate_protanomaly(image_array)
    elif type == 'option-3':
        image_array = simulate_protanopia(image_array)
    elif type == 'option-4':
        image_array = simulate_deuteranopia(image_array)
    elif type == 'option-5':
        image_array = simulate_tritanomaly(image_array)
    elif type == 'option-6':
        image_array = simulate_tritanopia(image_array)
    elif type == 'option-7':
        image_array = simulate_monochromacy(image_array)

    return image_array

def simulate_deuteranomaly(image_array):
    image_array[:, :, 1] = 0.5 * image_array[:, :, 1] + 0.5 * image_array[:, :, 0]
    return image_array

def simulate_protanomaly(image_array):
    image_array[:, :, 0] = 0.5 * image_array[:, :, 0] + 0.5 * image_array[:, :, 1]
    image_array *= 0.75
    return image_array

def simulate_protanopia(image_array):
    average_blue = np.mean(image_array[:, :, 2])
    image_array[:, :, 0] = average_blue
    image_array[:, :, 1] = average_blue
    return image_array

def simulate_deuteranopia(image_array):
    average_blue = np.mean(image_array[:, :, 2])
    image_array[:, :, 0] = average_blue
    image_array[:, :, 1] = average_blue
    return image_array

def simulate_tritanomaly(image_array):
    image_array[:, :, 2] *= 0.8
    return image_array

def simulate_tritanopia(image_array):
    average_red = np.mean(image_array[:, :, 0])
    image_array[:, :, 1] = average_red
    image_array[:, :, 2] = average_red
    return image_array

def simulate_monochromacy(image_array):
    grayscale_image = np.dot(image_array[..., :3], [0.2989, 0.5870, 0.1140])
    monochrome_image = np.stack((grayscale_image,) * 3, axis=-1)
    return monochrome_image   

@app.route('/ecb_cgallery', methods=['POST'])
def ecb_cgallery():
    if request.method == 'POST':
        blindness_type = request.form['options']
        folder_path = 'static/Assets/images/highres'
        image_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.jpg')]
        
        selected_images = random.sample(image_files, 10)
        processed_images = []
        processed_folder = 'static/processed_images/'
        
        if not os.path.exists(processed_folder):
            os.makedirs(processed_folder)
        
        for idx, image_path in enumerate(selected_images):
            image = cv2.imread(image_path)
            image_array = np.array(image)
            simulated_array = simulate_color_blindness(image_array, blindness_type)
            simulated_image = Image.fromarray(simulated_array)
            
            processed_image_path = os.path.join(processed_folder, f'processed_image_{idx}.jpg')
            simulated_image.save(processed_image_path)
            processed_images.append(processed_image_path)
        
        return render_template('ecb_cgallery.html', image_urls=processed_images)


@app.route('/image/<int:image_index>')
def full_screen_image(image_index):
    image_urls = request.args.get('urls').split(',')
    image_url = image_urls[image_index]
    return render_template('full_screen_image.html', image_url=image_url)

def load_color_dataset(dataset_path):
    color_dataset = {}
    with open(dataset_path, 'r') as file:
        for line in file:
            color_name, r, g, b = line.strip().split(',')
            color_dataset[(int(r), int(g), int(b))] = color_name
    return color_dataset

def annotate_colors(image_path, num_colors, color_dataset):
    image = cv2.imread(image_path)
    pixels = image.reshape(-1, 3)
    kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init=10)
    kmeans.fit(pixels)
    colors = kmeans.cluster_centers_.astype(int)
    labels = kmeans.labels_
    total_pixels = image.shape[0] * image.shape[1]
    pixel_percentages = [count / total_pixels for count in Counter(labels).values()]
    sorted_colors = sorted(zip(pixel_percentages, colors), reverse=True)
    annotated_image = image.copy()
    for pixel_percentage, color in sorted_colors:
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        mask[labels.reshape(image.shape[:2]) == sorted_colors.index((pixel_percentage, color))] = 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0 or cv2.contourArea(max(contours, key=cv2.contourArea)) < 100:
            continue
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        cv2.rectangle(annotated_image, (x, y), (x + w, y + h), color.tolist(), 2)
        closest_color_name = find_closest_color(color, color_dataset)
        cv2.putText(annotated_image, f"{closest_color_name} ({int(pixel_percentage * 100)}%)", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color.tolist(), 2)
    _, buffer = cv2.imencode('.jpg', annotated_image)
    annotated_image_base64 = base64.b64encode(buffer).decode('utf-8')
    return annotated_image_base64

def find_closest_color(rgb, color_dataset):
    min_dist = float('inf')
    closest_color_name = None
    for color, name in color_dataset.items():
        dist = np.linalg.norm(np.array(rgb) - np.array(color))
        if dist < min_dist:
            min_dist = dist
            closest_color_name = name
    return closest_color_name

@app.route('/cb')
def cb():
    return render_template('cb.html')

@app.route('/cbai')
def cbai():
    images_dir = 'static/Assets/images/highres'
    image_files = [os.path.join(images_dir, file) for file in os.listdir(images_dir) if file.endswith('.png') or file.endswith('.jpg')]
    selected_images = random.sample(image_files, 10)
    processed_images = []
    processed_folder = 'static/processed_images/'
    if not os.path.exists(processed_folder):
        os.makedirs(processed_folder)
    num_colors = 5
    dataset_path = 'static/Assets/colors-full.csv'
    color_dataset = load_color_dataset(dataset_path)
    for idx, image_path in enumerate(selected_images):
        annotated_image_path = annotate_colors(image_path, num_colors, color_dataset)
        processed_images.append(annotated_image_path)
    session['processed_images'] = processed_images
    processed_images = session.get('processed_images', [])
    return render_template('cbai_gallery.html', annotated_images=processed_images)



def simulate_vd(image, spherical_power, cylindrical_power, axis_orientation):
    simulated_image = image.copy()
    if spherical_power != 0:
        if spherical_power < 0:
            blur_kernel_size = int(abs(spherical_power) * 4 + 1)
            blur_kernel_size = blur_kernel_size if blur_kernel_size % 2 == 1 else blur_kernel_size + 1
            simulated_image = cv2.GaussianBlur(simulated_image, (blur_kernel_size, blur_kernel_size), 0)
        else:
            focal_length = 1 / spherical_power
            simulated_image = cv2.resize(simulated_image, None, fx=focal_length, fy=focal_length, interpolation=cv2.INTER_LINEAR)
            simulated_image = cv2.resize(simulated_image, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    
    if cylindrical_power != 0:
        kernel_size = int(abs(cylindrical_power) * 4 + 1)
        kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        sigma = kernel_size / 4
        lambd = kernel_size / 2
        gamma = 0.5
        kernel = cv2.getGaborKernel((kernel_size, kernel_size), sigma, np.deg2rad(axis_orientation), lambd, gamma, 0, ktype=cv2.CV_32F)
        kernel /= 1.5 * kernel.sum()
        simulated_image = cv2.filter2D(simulated_image, -1, kernel)
    
    return simulated_image

@app.route('/evd_rgallery')
def evd_rgallery():
    folder_path = 'static/Assets/images/highres'
    image_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.png') or file.endswith('.jpg')]
    selected_images = random.sample(image_files, 10)
    processed_images = []
    processed_folder = 'static/processed_images/'
    if not os.path.exists(processed_folder):
        os.makedirs(processed_folder)
    for idx, image_path in enumerate(selected_images):
        spherical_power = random.uniform(-10, 10)
        cylindrical_power = random.uniform(-5, 5)
        axis_orientation = random.uniform(0, 180) 
        image = cv2.imread(image_path)
        processed_image = simulate_vd(image, spherical_power, cylindrical_power, axis_orientation)
        processed_image_path = os.path.join(processed_folder, f'processed_image_{idx}.jpg')
        cv2.imwrite(processed_image_path, processed_image)
        processed_images.append(processed_image_path)
    return render_template('evd_rgallery.html', image_urls=processed_images)

@app.route('/evd_cgallery', methods=['POST'])
def evd_cgallery():
   if request.method == 'POST':
        spherical_power = float(request.form['spherical'])
        cylindrical_power = float(request.form['cylindrical'])
        axis_orientation = float(request.form['axis'])
        folder_path = 'static/Assets/images/highres'
        image_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.png') or file.endswith('.jpg')]
        selected_images = random.sample(image_files, 10)
        processed_images = []
        processed_folder = 'static/processed_images/'
        if not os.path.exists(processed_folder):
            os.makedirs(processed_folder)
        for idx, image_path in enumerate(selected_images):
            image = cv2.imread(image_path)
            processed_image = simulate_vd(image, spherical_power, cylindrical_power, axis_orientation)
            processed_image_path = os.path.join(processed_folder, f'processed_image_{idx}.jpg')
            cv2.imwrite(processed_image_path, processed_image)
            processed_images.append(processed_image_path)
        return render_template('evd_rgallery.html', image_urls=processed_images)


def apply_correction(image, spherical_power, cylindrical_power, axis_orientation):
    height, width = image.shape[:2]
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    spherical_power *= -0.001 
    cylindrical_power *= -0.001
    cylindrical_power_rad = cylindrical_power * 2 * np.pi
    angle_rad = np.deg2rad(axis_orientation)
    x_distorted = x - width / 2
    y_distorted = y - height / 2

    x_corrected = x_distorted + spherical_power * x_distorted + cylindrical_power_rad * np.cos(2 * angle_rad) * x_distorted + cylindrical_power_rad * np.sin(2 * angle_rad) * y_distorted
    y_corrected = y_distorted + spherical_power * y_distorted + cylindrical_power_rad * np.sin(2 * angle_rad) * x_distorted + cylindrical_power_rad * np.cos(2 * angle_rad) * y_distorted

    x_corrected += width / 2
    y_corrected += height / 2

    corrected_image = cv2.remap(image, x_corrected.astype(np.float32), y_corrected.astype(np.float32), cv2.INTER_LINEAR)

    return corrected_image

@app.route('/vd_cgallery', methods=['POST'])
def vd_cgallery():
    if request.method == 'POST':
        spherical_power = float(request.form['spherical'])
        cylindrical_power = float(request.form['cylindrical'])
        axis_orientation = float(request.form['axis'])
        folder_path = 'static/Assets/images/highres'
        image_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.png') or file.endswith('.jpg')]
        selected_images = random.sample(image_files, 10)
        processed_images = []
        processed_folder = 'static/processed_images/'
        if not os.path.exists(processed_folder):
            os.makedirs(processed_folder)
        for idx, image_path in enumerate(selected_images):
            image = cv2.imread(image_path)
            processed_image = apply_correction(image, spherical_power, cylindrical_power, axis_orientation)
            processed_image_path = os.path.join(processed_folder, f'processed_image_{idx}.jpg')
            cv2.imwrite(processed_image_path, processed_image)
            processed_images.append(processed_image_path)
        return render_template('vd_cgallery.html', image_urls=processed_images)


if __name__ == "__main__":
    app.run(debug=True)

