from django.http import JsonResponse
from django.shortcuts import render
import face_recognition
import numpy as np
from PIL import Image
import base64
from io import BytesIO
import pytesseract
import re

def extract_passport_data(image):
    # Extract text using OCR
    ocr_result = pytesseract.image_to_string(image)

    # Define regex patterns for passport data
    passport_data = {
        "seria": None,
        "birthdate": None,
        "jshr": None,
    }

    # Seria (AABB1234567 format)
    seria_match = re.search(r'\b[A-Z]{2}\d{7}\b', ocr_result)
    if seria_match:
        passport_data["seria"] = seria_match.group(0)

    # Birthdate (DD.MM.YYYY or YYYY.MM.DD format)
    birthdate_match = re.search(r'\b\d{2}.\d{2}.\d{4}\b', ocr_result)  # DD.MM.YYYY
    if not birthdate_match:
        birthdate_match = re.search(r'\b\d{4}.\d{2}.\d{2}\b', ocr_result)  # YYYY.MM.DD
    if birthdate_match:
        passport_data["birthdate"] = birthdate_match.group(0)

    # JSHR (14-digit number)
    jshr_match = re.search(r'\b\d{14}\b', ocr_result)
    if jshr_match:
        passport_data["jshr"] = jshr_match.group(0)

    return passport_data

def split_and_compare_faces(image_data):
    # Decode base64 image data
    image_data = image_data.split(',')[1]
    image_bytes = base64.b64decode(image_data)
    image = Image.open(BytesIO(image_bytes))

    # Convert image to RGB
    image = image.convert("RGB")

    width, height = image.size
    left_half = image.crop((0, 0, width // 2, height))
    right_half = image.crop((width // 2, 0, width, height))

    # Encode faces
    left_encoding = face_recognition.face_encodings(np.array(left_half))
    right_encoding = face_recognition.face_encodings(np.array(right_half))

    if not left_encoding or not right_encoding:
        return None

    # Find face locations
    left_face_location = face_recognition.face_locations(np.array(left_half))
    right_face_location = face_recognition.face_locations(np.array(right_half))

    if not left_face_location or not right_face_location:
        return None

    left_face_area = (left_face_location[0][2] - left_face_location[0][0]) * (left_face_location[0][1] - left_face_location[0][3])
    right_face_area = (right_face_location[0][2] - right_face_location[0][0]) * (right_face_location[0][1] - right_face_location[0][3])

    # Identify passport face
    if left_face_area < right_face_area:
        passport_image = left_half
        passport_encoding = left_encoding[0]
        other_encoding = right_encoding[0]
    else:
        passport_image = right_half
        passport_encoding = right_encoding[0]
        other_encoding = left_encoding[0]

    # Compare faces
    match = face_recognition.compare_faces([passport_encoding], other_encoding)

    if match[0]:
        passport_data = extract_passport_data(passport_image)
        return {"match": True, "passport_data": passport_data}
    else:
        return {"match": False}

def index(request):
    if request.method == "POST" and request.POST.get("image"):
        image_data = request.POST["image"]
        match = split_and_compare_faces(image_data)
        if match is None:
            return JsonResponse({"error": "Yuzlar topilmadi yoki rasmda biror yuz topilmadi"})
        elif match["match"]:
            return JsonResponse({"message": "Ikkala yuz bir xil odamga tegishli!", "passport_data": match["passport_data"]})
        else:
            return JsonResponse({"error": "Bu boshqa odam yoki boshqattan urinib ko'ring!"})
    return render(request, "index.html")
