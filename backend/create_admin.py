#!/usr/bin/env python
import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from django.contrib.auth.models import User

# Create or update admin user
user, created = User.objects.get_or_create(username='admin')
user.email = 'admin@example.com'
user.is_staff = True
user.is_superuser = True
user.set_password('admin123')
user.save()

status = 'created' if created else 'updated'
print(f'Admin user {status}')
print(f'Username: admin')
print(f'Password: admin123')
