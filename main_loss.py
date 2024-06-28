import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# Function to load images from a folder
def load_images(folder_path):
    images = []
    filenames = []  # Store filenames for matching labels later
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg"):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            if img is not None:
                images.append(img)
                filenames.append(filename)  # Store filename
            else:
                print(f"Warning: Unable to load image {filename}")
    return images, filenames


# Function to convert images to grayscale
def convert_to_grayscale(images):
    gray_images = []
    for img in images:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_images.append(gray_img)
    return gray_images


# Define the folder path containing images
folder_path = "project/char_tmpl/"
# Load images and convert them to grayscale
images, filenames = load_images(folder_path)
gray_images = convert_to_grayscale(images)


# Function to denoise images using methods like mean, gaussian, median, bilateral
def denoise_images(images, method='gaussian'):
    denoised_images = []
    for img in images:
        if method == 'mean':
            denoised_img = cv2.blur(img, (5, 5))
        elif method == 'gaussian':
            denoised_img = cv2.GaussianBlur(img, (5, 5), 0)
        elif method == 'median':
            denoised_img = cv2.medianBlur(img, 5)
        elif method == 'bilateral':
            denoised_img = cv2.bilateralFilter(img, 9, 75, 75)
        else:
            raise ValueError("Unsupported method. Choose from 'mean', 'gaussian', 'median', or 'bilateral'.")
        denoised_images.append(denoised_img)
    return denoised_images

# Choose the denoising method
denoising_method = 'bilateral'
denoised_images = denoise_images(gray_images, method=denoising_method)

# Function to display denoised images
def display_images(images, title=''):
    for i, img in enumerate(images):
        plt.figure(figsize=(5, 5))
        plt.imshow(img, cmap='gray')
        plt.title(f'{title} Image {i}')
        plt.axis('off')
        plt.show()
#display_images(denoised_images, title=f'Denoised ({denoising_method.capitalize()})')

# Function to binarize images (convert to binary: 0 or 255)
def binarize_images(images, threshold=127):
    binarized_images = []
    for img in images:
        _, binarized_img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
        binarized_images.append(binarized_img)
    return binarized_images


# Binarize denoised images
binarized_images = binarize_images(denoised_images)


# Function to extract SIFT features from images
def extract_sift_features(images):
    sift = cv2.SIFT_create()
    keypoints_list = []
    descriptors_list = []

    for img in images:
        keypoints, descriptors = sift.detectAndCompute(img, None)
        keypoints_list.append(keypoints)
        descriptors_list.append(descriptors)

    return keypoints_list, descriptors_list


# Extract SIFT features from binarized images
keypoints_list, descriptors_list = extract_sift_features(binarized_images)


# Function to filter out images without any SIFT descriptors
def filter_images_with_descriptors(images, descriptors_list, filenames):
    filtered_images = []
    filtered_descriptors_list = []
    filtered_filenames = []  # Store the matching filenames

    for img, descriptors, filename in zip(images, descriptors_list, filenames):
        if descriptors is not None and len(descriptors) > 0:
            filtered_images.append(img)
            filtered_descriptors_list.append(descriptors)
            filtered_filenames.append(filename)  # Store filename of valid image

    return filtered_images, filtered_descriptors_list, filtered_filenames


# Filter out images without descriptors
filtered_images, filtered_descriptors_list, filtered_filenames = filter_images_with_descriptors(binarized_images,
                                                                                                descriptors_list,
                                                                                                filenames)

# Check the number of filtered images and descriptors
print("Number of filtered images:", len(filtered_images))
print("Number of filtered descriptors:", len(filtered_descriptors_list))


# Create a Bag of Visual Words (BoVW) representation
def create_bovw_features(descriptors_list, n_clusters=50):
    # Ensure all descriptors have the same size (128)
    valid_descriptors_list = [desc for desc in descriptors_list if desc is not None and desc.shape[1] == 128]

    if len(valid_descriptors_list) == 0:
        raise ValueError("No valid descriptors found.")

    # Stack all valid descriptors together for clustering
    all_descriptors = np.vstack(valid_descriptors_list)

    # Perform k-means clustering to create visual words
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(all_descriptors)

    # Create histograms of visual words for each image
    bovw_features = []
    for descriptors in descriptors_list:
        if descriptors is not None and descriptors.shape[1] == 128:
            # Predict the closest visual word for each descriptor
            words = kmeans.predict(descriptors)

            # Create histogram of visual words
            hist, _ = np.histogram(words, bins=np.arange(n_clusters + 1))
            bovw_features.append(hist)
        else:
            # If the descriptor size doesn't match, create a zero histogram
            bovw_features.append(np.zeros(n_clusters))

    return np.array(bovw_features), kmeans


# Create BoVW features
n_clusters = 10  # Number of visual words (clusters)
bovw_features, kmeans = create_bovw_features(filtered_descriptors_list, n_clusters=n_clusters)

# Check the number of BoVW features
print("Number of BoVW features:", bovw_features.shape[0])

# Normalize the BoVW features
scaler = StandardScaler()
bovw_features_normalized = scaler.fit_transform(bovw_features)

# Check the normalized BoVW features
print("Number of normalized BoVW features:", bovw_features_normalized.shape[0])

# Function to load labels from image filenames and match with filtered images
def load_labels_filtered(filenames):
    labels = []
    for filename in filenames:
        if filename.endswith(".jpg"):
            # Extract the digit from the filename
            label = int(filename.split('.')[0]) % 10  # Get the last digit of the filename
            labels.append(label)
    return labels


# Load labels for filtered images
filtered_labels = load_labels_filtered(filtered_filenames)

# Check the number of filtered labels
print("Number of filtered labels:", len(filtered_labels))

# Ensure number of features and labels match
if len(bovw_features_normalized) != len(filtered_labels):
    raise ValueError(
        f"Number of BoVW features ({len(bovw_features_normalized)}) and filtered labels ({len(filtered_labels)}) do not match.")

# Apply PCA to normalized BoVW features
pca = PCA(n_components=0.5)
bovw_features_pca = pca.fit_transform(bovw_features_normalized)

# Check the shape of PCA-transformed features
print("Number of PCA-transformed BoVW features:", bovw_features_pca.shape[0])
print("Number of PCA components selected:", pca.n_components_)

# 可视化前两个主成分
def visualize_pca_2d(bovw_features_pca):
    plt.figure(figsize=(8, 6))
    plt.scatter(bovw_features_pca[:, 0], bovw_features_pca[:, 1], s=50, alpha=0.7, edgecolors='k')
    plt.title('PCA of BoVW Features (First Two Components)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()

visualize_pca_2d(bovw_features_pca)

# Split PCA-transformed data into train and test sets
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(bovw_features_pca, filtered_labels, test_size=0.2,
                                                                    random_state=42)

# Initialize models with PCA-transformed data
lda_pca = LDA()
svm_pca = SVC(kernel='linear',  probability=True)
logistic_regression_pca = LogisticRegression(max_iter=1000, penalty='l2', solver='lbfgs')

# Fit models with PCA-transformed data
lda_pca.fit(X_train_pca, y_train_pca)
svm_pca.fit(X_train_pca, y_train_pca)
logistic_regression_pca.fit(X_train_pca, y_train_pca)

# Predict using models with PCA-transformed data
lda_pred_pca = lda_pca.predict(X_test_pca)
svm_pred_pca = svm_pca.predict(X_test_pca)
logistic_regression_pred_pca = logistic_regression_pca.predict(X_test_pca)

# Evaluate models with PCA-transformed data
print("LDA Classification Report with PCA:")
print(classification_report(y_test_pca, lda_pred_pca))
print("LDA Accuracy with PCA:", accuracy_score(y_test_pca, lda_pred_pca))

print("SVM Classification Report with PCA:")
print(classification_report(y_test_pca, svm_pred_pca))
print("SVM Accuracy with PCA:", accuracy_score(y_test_pca, svm_pred_pca))

print("Logistic Regression Classification Report with PCA:")
print(classification_report(y_test_pca, logistic_regression_pred_pca))
print("Logistic Regression Accuracy with PCA:", accuracy_score(y_test_pca, logistic_regression_pred_pca))


# Function to predict on a single input image
def predict_single_image(image_path, scaler, pca, lda_model, svm_model, logistic_model, kmeans, n_clusters=50):
    # Load and preprocess the input image
    input_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if input_image is None:
        print(f"Error: Unable to load image {image_path}")
        return None

    # Denoise and binarize the input image (similar preprocessing as done with the dataset)
    denoised_img = cv2.bilateralFilter(input_image, 9, 75, 75)
    _, binarized_img = cv2.threshold(denoised_img, 127, 255, cv2.THRESH_BINARY)

    # Extract SIFT features
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(binarized_img, None)

    # If no descriptors found, return None
    if descriptors is None or len(descriptors) == 0:
        print("Error: No descriptors found in the input image.")
        return None

    # Create BoVW histogram for the input image
    words = kmeans.predict(descriptors)
    hist, _ = np.histogram(words, bins=np.arange(n_clusters + 1))
    bovw_features = np.array([hist])

    # Normalize BoVW features using the provided scaler
    bovw_features_normalized = scaler.transform(bovw_features)

    # Reduce dimensionality using PCA
    bovw_features_pca = pca.transform(bovw_features_normalized)

    # Predict using LDA, SVM, and Logistic Regression models
    lda_pred = lda_model.predict(bovw_features_pca)
    svm_pred = svm_model.predict(bovw_features_pca)
    logistic_pred = logistic_model.predict(bovw_features_pca)

    # Return predictions
    return lda_pred[0], svm_pred[0], logistic_pred[0]


# Provide a path to an image for prediction
input_image_path = "C:\pyprojects\lessons\pattern_detection\Project/plates/plate_10.jpg"
lda_prediction, svm_prediction, logistic_prediction = predict_single_image(input_image_path, scaler, pca, lda_pca,
                                                                           svm_pca, logistic_regression_pca, kmeans,
                                                                           n_clusters)

# Print predictions
print("LDA Prediction:", lda_prediction)
print("SVM Prediction:", svm_prediction)
print("Logistic Regression Prediction:", logistic_prediction)


"""detect the plate and segment the number, get the result
"""
# Step 1: Detect license plate using color and contour detection
def detect_license_plate(image):
    # Convert image to gray scale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Bilateral Filter to reduce noise while keeping edges sharp
    blurred = cv2.bilateralFilter(gray, 9, 75, 75)

    # Use adaptive threshold to highlight the plate area
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 19, 9)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    license_plate_contour = None
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) == 4:  # Looking for quadrilateral
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            area = cv2.contourArea(approx)
            if aspect_ratio > 2 and area > 1000:  # Filter out based on aspect ratio and area
                license_plate_contour = approx
                break

    if license_plate_contour is None:
        raise ValueError("No license plate found in the image.")

    return license_plate_contour


# Function to extract the license plate image using perspective transformation
def extract_license_plate_image(image, contour):
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.intp(box)

    width = int(rect[1][0])
    height = int(rect[1][1])
    src_pts = box.astype("float32")
    dst_pts = np.array([[0, height - 1],
                        [0, 0],
                        [width - 1, 0],
                        [width - 1, height - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(image, M, (width, height))

    return warped

# Function to segment characters from the license plate
def segment_characters(license_plate_image):
    gray = cv2.cvtColor(license_plate_image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    character_images = []
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        if w >= 10 and h >= 30:  # Filter out small contours that are not likely to be characters
            char_img = binary[y:y + h, x:x + w]
            char_img = cv2.resize(char_img, (20, 40), interpolation=cv2.INTER_AREA)
            character_images.append(char_img)

    # Sort character images from left to right
    character_images = sorted(character_images, key=lambda img: cv2.boundingRect(contours[character_images.index(img)])[0])

    return character_images



# Function to predict characters
def recognize_characters(character_images, scaler, pca, lda_model, svm_model, logistic_model, kmeans, n_clusters=50):
    recognized_chars = []
    for char_img in character_images:
        char_img = cv2.resize(char_img, (20, 20), interpolation=cv2.INTER_AREA)
        _, char_binarized = cv2.threshold(char_img, 127, 255, cv2.THRESH_BINARY)
        char_binarized = cv2.bitwise_not(char_binarized)

        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(char_binarized, None)

        if descriptors is not None and len(descriptors) > 0:
            words = kmeans.predict(descriptors)
            hist, _ = np.histogram(words, bins=np.arange(n_clusters + 1))
            bovw_features = np.array([hist])

            bovw_features_normalized = scaler.transform(bovw_features)
            bovw_features_pca = pca.transform(bovw_features_normalized)

            lda_pred = lda_model.predict(bovw_features_pca)[0]
            svm_pred = svm_model.predict(bovw_features_pca)[0]
            logistic_pred = logistic_model.predict(bovw_features_pca)[0]

            recognized_chars.append((lda_pred, svm_pred, logistic_pred))

    return recognized_chars


# Load an image with a license plate
input_image_path = "C:\pyprojects\lessons\pattern_detection\Project/real/real_1.jpg"
input_image = cv2.imread(input_image_path)

# Step 1: Detect the license plate
license_plate_contour = detect_license_plate(input_image)

# Draw the detected contour on the original image
detected_image = cv2.drawContours(input_image.copy(), [license_plate_contour], -1, (0, 255, 0), 3)

# Step 2: Extract the license plate image
warped_license_plate = extract_license_plate_image(input_image, license_plate_contour)

# Display the detected license plate
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Original Image with Detected License Plate')
plt.imshow(cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB))

plt.subplot(1, 2, 2)
plt.title('Warped License Plate')
plt.imshow(cv2.cvtColor(warped_license_plate, cv2.COLOR_BGR2RGB))
plt.show()

# Step 3: Segment characters from the license plate
character_images = segment_characters(warped_license_plate)

# Display segmented characters
for i, char_img in enumerate(character_images):
    plt.figure(figsize=(2, 2))
    plt.title(f'Character {i + 1}')
    plt.imshow(char_img, cmap='gray')
    plt.axis('off')
    plt.show()

# Step 4: Recognize characters using pre-trained models
recognized_characters = recognize_characters(character_images, scaler, pca, lda_pca, svm_pca, logistic_regression_pca,
                                             kmeans, n_clusters)

# Print predictions for each character
for i, (lda_pred, svm_pred, logistic_pred) in enumerate(recognized_characters):
    print(
        f"Character {i + 1} - LDA Prediction: {lda_pred}, SVM Prediction: {svm_pred}, Logistic Regression Prediction: {logistic_pred}")


"""segment the plate and get the result
"""
# Step 1: Detect license plate using color and contour detection
def detect_license_plate(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 19, 9)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    license_plate_contour = None
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            area = cv2.contourArea(approx)
            if aspect_ratio > 2 and area > 1000:
                license_plate_contour = approx
                break

    if license_plate_contour is None:
        raise ValueError("No license plate found in the image.")

    return license_plate_contour

# Function to extract the license plate image using perspective transformation
def extract_license_plate_image(image, contour):
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    width = int(rect[1][0])
    height = int(rect[1][1])
    src_pts = box.astype("float32")
    dst_pts = np.array([[0, height - 1],
                        [0, 0],
                        [width - 1, 0],
                        [width - 1, height - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(image, M, (width, height))

    return warped

# Function to segment characters from the license plate
def segment_characters(license_plate_image):
    gray = cv2.cvtColor(license_plate_image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    character_images = []
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        if w >= 10 and h >= 30:
            char_img = binary[y:y + h, x:x + w]
            char_img = cv2.resize(char_img, (20, 40), interpolation=cv2.INTER_AREA)
            character_images.append(char_img)

    character_images = sorted(character_images, key=lambda img: cv2.boundingRect(contours[character_images.index(img)])[0])

    return character_images

# Function to predict characters
def recognize_characters(character_images, scaler, pca, lda_model, svm_model, logistic_model, kmeans, n_clusters=50):
    recognized_chars = []
    for char_img in character_images:
        char_img = cv2.resize(char_img, (20, 20), interpolation=cv2.INTER_AREA)
        _, char_binarized = cv2.threshold(char_img, 127, 255, cv2.THRESH_BINARY)
        char_binarized = cv2.bitwise_not(char_binarized)

        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(char_binarized, None)

        if descriptors is not None and len(descriptors) > 0:
            words = kmeans.predict(descriptors)
            hist, _ = np.histogram(words, bins=np.arange(n_clusters + 1))
            bovw_features = np.array([hist])

            bovw_features_normalized = scaler.transform(bovw_features)
            bovw_features_pca = pca.transform(bovw_features_normalized)

            lda_pred = lda_model.predict(bovw_features_pca)[0]
            svm_pred = svm_model.predict(bovw_features_pca)[0]
            logistic_pred = logistic_model.predict(bovw_features_pca)[0]

            recognized_chars.append((lda_pred, svm_pred, logistic_pred))

    return recognized_chars

# Load the image
input_image_path = "C:\pyprojects\lessons\pattern_detection\Project\plates_image/1.jpg"
input_image = cv2.imread(input_image_path)

# Step 1: Detect the license plate
try:
    license_plate_contour = detect_license_plate(input_image)
    # Draw the detected contour on the original image
    detected_image = cv2.drawContours(input_image.copy(), [license_plate_contour], -1, (0, 255, 0), 3)

    # Step 2: Extract the license plate image
    warped_license_plate = extract_license_plate_image(input_image, license_plate_contour)

    # Display the detected license plate
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title('Original Image with Detected License Plate')
    plt.imshow(cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB))

    plt.subplot(1, 2, 2)
    plt.title('Warped License Plate')
    plt.imshow(cv2.cvtColor(warped_license_plate, cv2.COLOR_BGR2RGB))
    plt.show()

    # Step 3: Segment characters from the license plate
    character_images = segment_characters(warped_license_plate)

    # Display segmented characters
    for i, char_img in enumerate(character_images):
        plt.figure(figsize=(2, 2))
        plt.title(f'Character {i + 1}')
        plt.imshow(char_img, cmap='gray')
        plt.axis('off')
        plt.show()

    # Step 4: Recognize characters using pre-trained models
    recognized_characters = recognize_characters(character_images, scaler, pca, lda_pca, svm_pca, logistic_regression_pca, kmeans, n_clusters)

    # Print predictions for each character
    for i, (lda_pred, svm_pred, logistic_pred) in enumerate(recognized_characters):
        print(f"Character {i + 1} - LDA Prediction: {lda_pred}, SVM Prediction: {svm_pred}, Logistic Regression Prediction: {logistic_pred}")

except ValueError as e:
    print(e)

import numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class CustomLogisticRegression(BaseEstimator, ClassifierMixin):
    def __init__(self, penalty='l1', alpha=1.0, loss='cross', max_iter=1000, lambda_reg=0.1):
        self.penalty = penalty
        self.alpha = alpha
        self.loss = loss
        self.max_iter = max_iter
        self.lambda_reg = lambda_reg
        self.theta_ = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def loss_function(self, theta, X, y):
        m = len(y)
        h = self.sigmoid(X @ theta)

        # 确保所有变量都是 numpy 数组
        theta = np.array(theta)
        X = np.array(X)
        y = np.array(y)
        h = np.array(h)

        # 损失函数部分
        if self.loss == 'cross':
            loss = - (1/m) * (y @ np.log(h) + (1 - y) @ np.log(1 - h))

        # 正则化项
        if self.penalty == 'l2':
            reg = (self.alpha / (2 * m)) * np.sum(theta[1:] ** 2)
        elif self.penalty == 'l1':
            reg = (self.alpha / m) * np.sum(np.abs(theta[1:]))
        else:
            reg = 0
        return loss + self.lambda_reg * reg  # 使用 lambda_reg 进行加权


    def fit(self, X, y):
        X = np.insert(X, 0, 1, axis=1)  # 添加偏置项
        initial_theta = np.zeros(X.shape[1])

        opt_results = minimize(
            fun=self.loss_function,
            x0=initial_theta,
            args=(X, y),
            method='L-BFGS-B',
            options={'maxiter': self.max_iter}
        )

        self.theta_ = opt_results.x
        return self

    def predict_proba(self, X):
        X = np.insert(X, 0, 1, axis=1)  # 添加偏置项
        prob = self.sigmoid(X @ self.theta_)
        return np.column_stack([1 - prob, prob])

    def predict(self, X):
        prob = self.predict_proba(X)[:, 1]
        return (prob >= 0.5).astype(int)


# 使用 PCA 处理后的数据进行训练和评估

# 训练自定义逻辑回归模型
custom_logistic_regression = CustomLogisticRegression(penalty='l1', loss='cross', max_iter=1000, alpha=1.0,lambda_reg=0.1)
custom_logistic_regression.fit(X_train_pca, y_train_pca)

# 使用自定义逻辑回归模型进行预测
custom_logistic_pred_pca = custom_logistic_regression.predict(X_test_pca)

# 评估自定义逻辑回归模型
print("Custom Logistic Regression Classification Report with PCA:")
print(classification_report(y_test_pca, custom_logistic_pred_pca))
print("Custom Logistic Regression Accuracy with PCA:", accuracy_score(y_test_pca, custom_logistic_pred_pca))
