from django import forms


class TextLangForm(forms.Form):
    text = forms.CharField(max_length=1000000)
