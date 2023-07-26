from .models import Text
from django.forms import ModelForm, TextInput

class TextForm(ModelForm):
	class Meta:
		model = Text
		fields = ['name']
		widgets = {'name': TextInput(attrs={
			'class': 'form-control',
			'name': 'mov',
			'id': 'mov',
			'placeholder': 'Введите отзыв на английском языке'
			})}