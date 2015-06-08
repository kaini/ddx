#!/usr/bin/env python3
import os
import sys
import tweepy
from random import choice, seed
from collections import defaultdict
from datetime import datetime, date

INT_TYPES = ["Easy", "Medium", "Hard", "Special"]

if __name__ == "__main__":
    ckey = os.getenv("TWITTER_CKEY", "")
    skey = os.getenv("TWITTER_SKEY", "")
    authtoken = os.getenv("TWITTER_ATOKEN", "")
    authsecret = os.getenv("TWITTER_ASECRET", "")
    if not ckey or not skey or not authtoken or not authsecret:
        print("TWITTER_SKEY, TWITTER_CKEY, TWITTER_ATOKEN or TWITTER_ASECRET not provided")
        sys.exit(1)

    seed()

    integrals = defaultdict(list)
    for item in os.listdir("integrals"):
        for type in INT_TYPES:
            if item.lower().startswith(type.lower()) and item.lower().endswith(".png"):
                integrals[type].append("integrals/" + item)
                break
    if sum(len(l) for l in integrals.values()) <= 14:
        print("WARNING: only less than 14 integrals left")

    use_special = False
    today = date.today()
    filename = "%s_%04d-%02d-%02d_" % (INT_TYPES[-1], today.year,
                                       today.month, today.day)
    filename = filename.lower()
    for integral in integrals[INT_TYPES[-1]]:
        if integral.lower().startswith("integrals/" + filename):
            use_special = True
            status = integral.split(".")[0].split("_")[-1]
            break

    if not use_special:
        try:
            with open("integrals/lasttype.txt") as fp:
                lasttype = int(fp.read())
        except FileNotFoundError:
            lasttype = len(INT_TYPES) - 2  # ignore Special
        nexttype = (lasttype + 1) % (len(INT_TYPES) - 1)

        integral = choice(integrals[INT_TYPES[nexttype]])
        status = "Difficulty: " + INT_TYPES[nexttype]

        with open("integrals/lasttype.txt", "w") as fp:
            fp.write(str(nexttype))

    if False:  # DEBUG DEBUG
        print("Image:", integral, "Status:", status)
        sys.exit(1)

    auth = tweepy.OAuthHandler(ckey, skey)
    auth.set_access_token(authtoken, authsecret)
    api = tweepy.API(auth)
    with open(integral, "rb") as fp:
        api.update_with_media(integral,
                              status=status,
                              file=fp)

    os.rename(integral, integral + ".used." + str(int(datetime.utcnow().timestamp())))

