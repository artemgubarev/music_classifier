import telebot
import os
import tempfile
import threading
from predictor import predict_genre, get_audio_duration
from keras.models import load_model

telebot.apihelper.READ_TIMEOUT = 60
telebot.apihelper.CONNECT_TIMEOUT = 10

MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024

bot = telebot.TeleBot('7880510286:AAE_aw_wEY__TCQPsOF9Cg9FSrx4v2Ou5VA')

SUPPORTED_AUDIO_MIME_TYPES = {
    'audio/mpeg', 'audio/wav', 'audio/x-wav', 'audio/vnd.wave', 'audio/x-aiff',
    'audio/flac', 'audio/mp4', 'audio/aac', 'audio/ogg', 'audio/x-ms-wma',
    'audio/midi', 'audio/x-midi', 'audio/amr', 'audio/ac3', 'audio/x-dsf',
    'audio/opus', 'audio/x-opus', 'application/x-mqa',
}

MIN_DURATION_SECONDS = 4
model = load_model("genre_classification_cnn.h5")

user_states = set()

def process_file(message, file_info):
    try:
        if file_info.file_size > MAX_FILE_SIZE_BYTES:
            bot.send_message(
                message.chat.id,
                "❗️ Файл слишком большой (> 20MB). Отправьте, пожалуйста, файл поменьше."
            )
            return
        
        file = bot.get_file(file_info.file_id)
        downloaded = bot.download_file(file.file_path)

        ext = os.path.splitext(file.file_path)[1] or ".mp3"
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(downloaded)
            tmp_path = tmp.name

        duration = get_audio_duration(tmp_path)
        if duration < MIN_DURATION_SECONDS:
            bot.send_message(
                message.chat.id,
                f"⚠️ Композиция слишком короткая ({duration:.2f} сек). Нужно минимум {MIN_DURATION_SECONDS} сек."
            )
            return

        result = predict_genre(model, tmp_path)
        bot.send_message(message.chat.id, result)

    except Exception as e:
        bot.send_message(message.chat.id, f"❌ Ошибка при обработке: {str(e)}")

    finally:
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.remove(tmp_path)

@bot.message_handler(commands=['start'])
def start_handler(message):
    user_states.add(message.chat.id)
    bot.send_message(message.chat.id, "👋 Бот активирован. Отправьте музыкальный файл.")

@bot.message_handler(content_types=['audio'])
def handle_audio(message):
    if message.chat.id not in user_states:
        bot.reply_to(message, "🔒 Пожалуйста, начните работу с командой /start.")
        return
    bot.reply_to(message, "🎧 Файл получен. Определяю жанр...")
    threading.Thread(target=process_file, args=(message, message.audio)).start()

@bot.message_handler(content_types=['document'])
def handle_document(message):
    if message.chat.id not in user_states:
        bot.reply_to(message, "🔒 Пожалуйста, начните работу с командой /start.")
        return
    mime_type = message.document.mime_type
    if mime_type in SUPPORTED_AUDIO_MIME_TYPES:
        bot.reply_to(message, "🎧 Файл получен. Определяю жанр...")
        threading.Thread(target=process_file, args=(message, message.document)).start()
    else:
        bot.reply_to(message, f"❗️Файл типа `{mime_type}` не поддерживается.")

@bot.message_handler(func=lambda m: True, content_types=[
    'text', 'photo', 'video', 'voice', 'sticker', 'location', 'contact',
    'animation', 'video_note', 'poll', 'dice', 'venue'
])
def block_everything_else(message):
    if message.chat.id not in user_states:
        bot.reply_to(message, "🔒 Пожалуйста, начните работу с командой /start.")
    else:
        bot.reply_to(message, "❗️Я принимаю только музыкальные файлы. Пожалуйста, отправьте аудио.")
        
bot.polling()