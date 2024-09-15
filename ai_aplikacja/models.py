from django.core.files.storage import default_storage
from tensorflow.keras.preprocessing import image as tf_image
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
from django.core.files.base import ContentFile
import numpy as np


from django.db import models  # Importuje klasy i metody Django do pracy z modelami bazy danych
import openai  # Importuje bibliotekę klienta OpenAI do interakcji z API OpenAI




#klucz API OpenAI tutaj
openai.api_key = ''  # Pobiera wartość zmiennej środowiskowej 'OPENAI_API_KEY' i przypisuje ją jako klucz API do klienta OpenAI


def create_description(title):
    try:
        completion = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "user",
                    "content": f"Generate a descriptive text based on the following title:  {title}\n",
                    "max_tokens": "10"
                },
            ],
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        # Ogarnia errory i daje stosowną wiadomość
        print(f"Error generating description: {str(e)}")
        return f"Error: {str(e)}"

class TextElement(models.Model):  # Definiuje klasę TextElement jako model Django
    title = models.CharField(max_length=255)  # Pole 'title' typu CharField, przechowujące tytuł elementu tekstu
    content = models.TextField(blank=True)  # Pole 'content' typu TextField, przechowujące treść elementu tekstu; może być puste

    def __str__(self):  # Metoda specjalna, definiująca reprezentację tekstową instancji modelu
        return self.title  # Zwraca tytuł elementu tekstu jako jego reprezentację tekstową

    def save(self, *args, **kwargs):  # Nadpisuje metodę save, aby dostosować proces zapisu instancji modelu
        # Generuj opis tylko jeśli content jest pusty
        if not self.content:  # Sprawdza, czy pole 'content' jest puste
            self.content = create_description(self.title)  # Jeśli tak, wywołuje funkcję generate_description_from_title, aby wygenerować i przypisać treść
        super().save(*args, **kwargs)  # Wywołuje oryginalną metodę save klasy nadrzędnej, aby zapisać zmiany w instancji modelu


class ImageElement(models.Model):
    title = models.CharField(max_length=100, blank=True)
    content = models.TextField(blank=True)
    photo = models.ImageField(upload_to='mediaphoto', blank=True, null=True)

    def __str__(self):
        return self.title

    def save(self, *args, **kwargs):
        """
        Ladowanie obrazu o zadanych wymierach,
        konwertowanie obrazu na tablicę np,
        rozszerzanie tablicy o nowy wymiar,
        Przetwarzanie obrazu zgodnie z wymaganiami modelu
        Zapisywanie obrazu z predykcjami i z najprawdopodobniejszym tytułem

        :param args:
        :param kwargs:
        :return:
        """
        super().save(*args, **kwargs)

        if self.photo:
            try:
                file_path = self.photo.path
                if default_storage.exists(file_path):

                    pill_image = tf_image.load_img(file_path, target_size=(299, 299))

                    img_array = tf_image.img_to_array(pill_image)
                    img_array = np.expand_dims(img_array, axis=0)
                    img_array = preprocess_input(img_array)

                    model = InceptionV3(weights='imagenet')
                    prediction = model.predict(img_array)
                    decoded_prediction = decode_predictions(prediction, top=1)[0]
                    best_guess = decoded_prediction[0][1]
                    self.title = best_guess
                    self.content = ', '.join([f"{pred[1]}: {pred[2] * 100:.2f}%" for pred in decoded_prediction])
                    super().save(*args, **kwargs)

            except Exception as e:
                print(e)
                pass