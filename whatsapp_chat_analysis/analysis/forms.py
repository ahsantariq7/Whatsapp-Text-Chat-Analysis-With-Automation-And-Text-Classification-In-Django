from django import forms

from .models import UserSearch


class UserSearchForm(forms.ModelForm):
    class Meta:
        model = UserSearch
        fields = "__all__"
