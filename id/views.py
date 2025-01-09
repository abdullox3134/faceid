import base64
from django.shortcuts import render
import numpy as np
from PIL import Image
from io import BytesIO
from django.http import JsonResponse
import face_recognition
import pytesseract
import re

def extract_passport_data(image):
    ocr_result = pytesseract.image_to_string(image)
    passport_data = {
        "seria": None,
        "birthdate": None,
        "jshr": None,
    }
    seria_match = re.search(r'\b[A-Z]{2}\d{7}\b', ocr_result)
    if seria_match:
        passport_data["seria"] = seria_match.group(0)
    birthdate_match = re.search(r'\b\d{2}.\d{2}.\d{4}\b', ocr_result) or re.search(r'\b\d{4}.\d{2}.\d{2}\b', ocr_result)
    if birthdate_match:
        passport_data["birthdate"] = birthdate_match.group(0)
    jshr_match = re.search(r'\b\d{14}\b', ocr_result)
    if jshr_match:
        passport_data["jshr"] = jshr_match.group(0)
    return passport_data

def compare_two_faces(face_data, passport_data):
    face_image = Image.open(BytesIO(base64.b64decode(face_data.split(",")[1]))).convert("RGB")
    passport_image = Image.open(BytesIO(base64.b64decode(passport_data.split(",")[1]))).convert("RGB")

    face_encoding = face_recognition.face_encodings(np.array(face_image))
    passport_encoding = face_recognition.face_encodings(np.array(passport_image))

    if not face_encoding or not passport_encoding:
        return None

    match = face_recognition.compare_faces([face_encoding[0]], passport_encoding[0])
    if match[0]:
        passport_info = extract_passport_data(passport_image)
        return {"match": True, "passport_data": passport_info}
    return {"match": False}

def index(request):
    if request.method == "POST":
        face_image = request.POST.get("face_image")
        passport_image = request.POST.get("passport_image")
        if face_image and passport_image:
            result = compare_two_faces(face_image, passport_image)
            if result is None:
                return JsonResponse({"error": "Yuzlar topilmadi yoki rasmda biror yuz topilmadi"})
            elif result["match"]:
                return JsonResponse({"message": "Ikkala yuz bir xil odamga tegishli!", "passport_data": result["passport_data"]})
            else:
                return JsonResponse({"error": "Bu boshqa odam yoki boshqattan urinib ko'ring!"})
        return JsonResponse({"error": "Ikkala rasm ham kerak"})
    return render(request, "index.html")
