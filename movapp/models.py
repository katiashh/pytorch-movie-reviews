from django.db import models

class Text(models.Model):
	name = models.CharField(max_length=1000000)

	def __str__(self):
		return self.name
