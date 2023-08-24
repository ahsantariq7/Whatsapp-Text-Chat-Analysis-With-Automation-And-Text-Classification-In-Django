from django.db import models


# Create your models here.
class UserSearch(models.Model):
    username = models.CharField(max_length=200)
    range_scroll = models.PositiveIntegerField()
