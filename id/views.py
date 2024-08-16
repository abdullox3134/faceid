from django.http import JsonResponse
from django.shortcuts import render
import face_recognition
import numpy as np
from PIL import Image
import base64
from io import BytesIO


def split_and_compare_faces(image_data):
    # Decode the base64 image data
    image_data = image_data.split(',')[1]  # Remove the 'data:image/png;base64,' part
    image_bytes = base64.b64decode(image_data)
    image = Image.open(BytesIO(image_bytes))

    # Convert the image to RGB format (face_recognition requires 8-bit gray or RGB)
    image = image.convert("RGB")

    width, height = image.size
    left_half = image.crop((0, 0, width // 2, height))
    right_half = image.crop((width // 2, 0, width, height))

    # Yuzlarni encoding qilish
    left_encoding = face_recognition.face_encodings(np.array(left_half))
    right_encoding = face_recognition.face_encodings(np.array(right_half))

    # Agar biror bo'lakda yuz topilmasa
    if not left_encoding or not right_encoding:
        return None

    # Yuzlarni aniqlash va kattaligini hisoblash
    left_face_location = face_recognition.face_locations(np.array(left_half))
    right_face_location = face_recognition.face_locations(np.array(right_half))

    if not left_face_location or not right_face_location:
        return None

    left_face_area = (left_face_location[0][2] - left_face_location[0][0]) * (
                left_face_location[0][1] - left_face_location[0][3])
    right_face_area = (right_face_location[0][2] - right_face_location[0][0]) * (
                right_face_location[0][1] - right_face_location[0][3])

    # Pasport yuzini aniqlash (kichikroq yuz)
    if left_face_area < right_face_area:
        passport_encoding = left_encoding[0]
        other_encoding = right_encoding[0]
    else:
        passport_encoding = right_encoding[0]
        other_encoding = left_encoding[0]

    # Yuzlarni solishtirish
    match = face_recognition.compare_faces([passport_encoding], other_encoding)
    return match[0]


def index(request):
    if request.method == "POST" and request.POST.get("image"):
        image_data = request.POST["image"]
        match = split_and_compare_faces(image_data)
        if match is None:
            return JsonResponse({"error": "Yuzlar topilmadi yoki rasmda biror yuz topilmadi"})
        elif match:
            return JsonResponse({"message": "Ikkala yuz bir xil odamga tegishli!"})
        else:
            return JsonResponse({"error": "Bu boshqa odam yoki boshqattan urinib ko'ring!"})
    return render(request, "index.html")
