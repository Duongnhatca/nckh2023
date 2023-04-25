import telegram
import asyncio
async def other_coro():
    bot = telegram.Bot(token="6211404922:AAEBn2rI4mm92avEXpoao_xPUZpsK6NMHVg")
    chat_id = "5243841729"
    await bot.send_message(chat_id=chat_id, text='Nội dung tin nhắn của bạn')
asyncio.run(other_coro())