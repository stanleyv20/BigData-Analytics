import tweepy
from tweepy import OAuthHandler
from tweepy import Stream
from tweepy.streaming import StreamListener
import socket
import json
from kafka import KafkaProducer

producer = KafkaProducer(bootstrap_servers='localhost:9092') #Same port as Kafka server
topic_name = "ElonTweets"

consumer_key = ""
consumer_secret = ""
access_token = ""
access_secret = ""

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)
api = tweepy.API(auth, wait_on_rate_limit=True)


class TweetsListener(StreamListener):
  def on_status(self, data):
      try:
          print(data.text)
          producer.send(topic_name, data.text.encode('utf-8'))
          return True
      except BaseException as e:
          print("Error on_data: %s" % str(e))
      return True
  def on_error(self, status):
      print(status)
      return True

def sendData():
  auth = OAuthHandler(consumer_key, consumer_secret)
  auth.set_access_token(access_token, access_secret)
  twitter_stream = Stream(auth, TweetsListener())
  twitter_stream.filter(track=["Elon"], stall_warnings=True, languages= ["en"])



sendData()

