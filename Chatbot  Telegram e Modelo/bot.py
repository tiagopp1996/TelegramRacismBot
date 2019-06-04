import csv
from telegram.ext import Updater, CommandHandler
import os
import subprocess

#leitura de retorno da classificação
def readTextCSV():

    #arquivo exportado pelo modelo com o resultado da classificação
    f= open("/home/tiago/Área de Trabalho/TextClassification-master/runs/1556972257/predictions.csv", "r")
    reader = csv.reader(f)
    for row in reader:
        original=row[0]
        classificacao=row[1]

    #print(original + "\n\n")
    #print(classificacao)
    return classificacao

#Gravação de frase para classificação
def writeTextCSV(value):

    #Arquivo contendo a frase inputada no telegram
    download_dir = "/home/tiago/Área de Trabalho/TextClassification-master/data/data.csv" #where you want the file to be downloaded to

    csv = open(download_dir, "w")
    #"w" indicates that you're writing strings to the file

    columnTitleRow = "label,content\n"
    csv.write(columnTitleRow)
    csv.write("0,"  + value)
    csv.close()


def main():
    updater = Updater('821747918:AAE0lJqDDHbaj1LQOaUxfunKHtSk84H4vGs')


    def hello(bot, update):
        update.message.reply_text(
            'Hello {}'.format(update.message.from_user.first_name))



    def racism(bot, update):
        update.message.reply_text("Aguarde a Classificação\n")
        Imput=update.message.text
        Imput=Imput.replace("/racismo","")
        writeTextCSV(Imput)
        #alterar caminhos para a execução
        os.system("python3 /home/tiago/'Área de Trabalho'/TextClassification-master/test2.py --test_data_file=/home/tiago/'Área de Trabalho'/TextClassification-master/data/data.csv --run_dir=/home/tiago/'Área de Trabalho'/TextClassification-master/runs/1556972257 --checkpoint=clf-4000")
        if(readTextCSV()=="0.0"):
            retornoPrediction="não é racismo"
        else:
            retornoPrediction="é racismo"

        update.message.reply_text("A frase:'" + Imput + "' " + retornoPrediction)




    updater.dispatcher.add_handler(CommandHandler('hello', hello))
    updater.dispatcher.add_handler(CommandHandler('racismo', racism))

    updater.start_polling()
    updater.idle()


main()
