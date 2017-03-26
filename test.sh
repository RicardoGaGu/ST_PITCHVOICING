#!/bin/bash

g++ pitch_compare.cpp -o pitch_compare

python pitch.py pda_ue.gui

./pitch_compare pda_ue.gui
