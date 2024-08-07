import dill
from apscheduler.schedulers.blocking import BlockingScheduler
import tzlocal
from datetime import datetime
import pandas as pd

sched = BlockingScheduler(timezone=tzlocal.get_localzone_name())

df = pd.read_csv('model/data/homework.csv')
with open('model/cars_pipe.pkl', 'rb') as file:
    model = dill.load(file)



@sched.scheduled_job('cron', second='*/5')
def on_time():
    data = df.sample(5)
    data['predicted_price_cat'] = model['model'].predict(data)
    print(data[['id', 'price', 'predicted_price_cat']])

if __name__ == '__main__':
    sched.start()