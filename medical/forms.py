from django import forms
from .models import UserInput


# DataFlair #File_Upload
class InputForm(forms.Form):
    transcription = forms.CharField()
    csv = forms.FileField()