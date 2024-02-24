from chat import Bot

bot = Bot()
bot.load_PDF_doc("documents/temp.pdf")
bot.retreval()
query = "what is this document?"
bot.chat(query)
