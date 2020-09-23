from io import StringIO
import os
from django.conf import settings

from django.shortcuts import render,redirect
from django.urls import reverse
from models.med_spec_detector import ModelClass
from django.core.files.storage import default_storage
from django.http import HttpResponse
import pandas as pd

last_csv_file_name = ""


def home(request):
    return render(request, 'medical/home.html')


def details(request):
    if request.method == 'POST':
        transcription = request.POST.get("transcription")
        modelclass = ModelClass()
        hasfile = False
        results = None
        file_data = None
        try:
            csv_file = request.FILES['csv']
            global last_csv_file_name
            last_csv_file_name = csv_file.name
            file_data = request.FILES['csv'].read().decode('utf-8')
            transcription = pd.read_csv(StringIO(file_data)).transcription.tolist()[0]
            # save_csv(csv_file)
            hasfile = True
        except:
            transcription = transcription
        if len(transcription) != 0 and hasfile == False:  # transcription evaluation
            results = modelclass.predict(transcription)

        if hasfile:  # file evaluation
            df = pd.read_csv(StringIO(file_data))
            # print(df)
            results = modelclass.file_prediction(df.transcription.tolist())
            df['result'] = results
            # print(df)
            BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

            df.to_csv(BASE_DIR + '/media/' + last_csv_file_name)

        if len(transcription) != 0:
            return render(request, 'medical/details.html', {'result': results, "hasfile": hasfile})

    return redirect(reverse('home'))


def save_csv(csv_file):
    default_storage.save(csv_file.name, csv_file)

# Simple CSV Write Operation


def download_csv(response):
    try:
        file = default_storage.open(last_csv_file_name)
        response = HttpResponse(file.read(), content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename=predicted_'+last_csv_file_name
        delete_csv(last_csv_file_name)
    except:
        return redirect(reverse('home'))
    return response


def delete_csv(file_name):
    os.remove(settings.BASE_DIR + '/media/' + file_name)