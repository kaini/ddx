#!/usr/bin/env python3
import subprocess
import sys
import io
import urllib.request
import urllib.parse
from random import randint, choice

def num(allow0=False):
    n = randint(-10, 10)
    if n == -10:
        return "(e)"
    elif n == 10:
        return "(pi)"
    elif not allow0 and n == 0:
        return num(False)
    else:
        return "(" + str(n) + ")"

def simpleterm():
    type = randint(0, 1)
    if type == 0:
        # a*fn(b*x)
        return "(" + num() + "*" + choice(["ln", "cos", "sin", "e^"]) + "(" + num() + "*x))"
    elif type == 1:
        # a*x^b
        return "(" + num() + "*x^" + num() + ")"

def simpleterm_offset():
    type = randint(0, 1)
    if type == 0:
        # a*fn(b*x+c)
        return "(" + num() + "*" + choice(["ln", "cos", "sin", "e^"]) + "(" + num() + "*x+" + num(True) + "))"
    elif type == 1:
        # a*(x+c)^b
        return "(" + num() + "*(x+" + num(True) + ")^" + num() + ")"
    
def simple():
    type = randint(0, 2)
    if type == 0:
        return "(" + simpleterm() + choice(["+", "-"]) + simpleterm() + ")"
    elif type == 1:
        return simpleterm_offset()
    elif type == 2:
        # a*fn(b*fn(c*x))
        return "(" + num() + "*" + choice(["ln", "cos", "sin", "e^"]) + "(" + \
            num() + "*" + choice(["ln", "cos", "sin", "e^"]) + "(" + \
            num() + "*" + "x)))"

def medium():
    type = randint(0, 2)
    if type == 0:
        return "(" + simple() + choice(["*"]) + simple() + ")"
    elif type == 1:
        return "(" + simple() + "^" + num() + ")"
    else:
        return None

# TODO difficulty
start = "(ln(x^2+x)*sin(x^3-x^2)*e^x)^x"

with open("input.txt", "w") as fp:
    fp.write("""
        [ddx].
        main_print(""" + start + """), !.
        halt.
    """)

with open("input.txt", "r") as infp:
    with open("output.txt", "w") as outfp:
        subprocess.call(["swipl", "--quiet", "ddx.pl"],
                        stdin=infp, stdout=outfp, stderr=outfp)

with open("output.txt", "r") as fp:
    output = fp.read()
latex = [l for l in output.split("\n") if l.startswith("\\int")][0]

get_arg = urllib.parse.quote("\\dpi{200} \\bg_white " + latex, safe="")
url = "http://latex.codecogs.com/png.latex?" + get_arg
with urllib.request.urlopen(url) as http:
    png_data = http.read()

with open("output.png", "wb") as fp:
    fp.write(png_data)
