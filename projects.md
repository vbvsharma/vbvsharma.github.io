---
layout: default
title: Projects
permalink: /projects/
---

# Projects

<br>
## Twitter Retweet Web Application
**Used:** Python, Django, Redis, PostgreSQL, Celery, Tweepy
<br>
**Description:** It is a web application which retweets a particular account's tweets, on the registered user's behalf automatically.
<br>
This application is integrated to a particular Twitter account. This Twitter account will be polled at frequent intervals (say, every 1 min, however, it is configurable) to look for new tweets. Once, we get a list of new tweets, a scheduler is set to retweet, from the registered users' account, after a certain time elapse (say, 1 hour, this is configurable in settings too). On the other hand, we need user's consent to retweet on their behalf. Hence, while the user registers himself/herself on the web application, the user authenticates the web application, via Twitter OAuth 2, to retweet on user's stead.
<br>
**Application:** This application was developed on a request from my colleague. He needed such an application to promote his business on Twitter. Hence, it can be used by others for similar purposes.