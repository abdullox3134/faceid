from django.http import JsonResponse
from django.shortcuts import render
import face_recognition
import numpy as np


def compare_faces(image1_file, image2_file):
    # Har ikkala rasmni yuklash
    image1 = face_recognition.load_image_file(image1_file)
    image2 = face_recognition.load_image_file(image2_file)

    # Rasm ichidagi yuzlarni encoding qilish
    image1_encoding = face_recognition.face_encodings(image1)
    image2_encoding = face_recognition.face_encodings(image2)

    # Agar biror rasmda yuz topilmasa
    if not image1_encoding or not image2_encoding:
        return False

    # Yuzlarni solishtirish
    matches = face_recognition.compare_faces([image1_encoding[0]], image2_encoding[0])
    return matches[0]


def index(request):
    if request.method == "POST" and request.FILES.get("image1") and request.FILES.get("image2"):
        image1_file = request.FILES["image1"]
        image2_file = request.FILES["image2"]
        match = compare_faces(image1_file, image2_file)
        if match:
            return JsonResponse({"message": "Muvofaqqiyatli tanildi!"})
        else:
            return JsonResponse({"error": "Bu boshqa odam yoki boshqattan urinib ko'ring!"})
    return render(request, "index.html")
