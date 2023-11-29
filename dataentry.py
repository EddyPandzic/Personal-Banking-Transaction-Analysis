# dataentry.py
# Author: Eddy Pandzic
# Description: Python application to enter banking data manually into Excel

import PySimpleGUI as sg
import pandas as pd

#Adding a theme to the GUI Application
sg.theme("DarkAmber")

file = "2019-2022_Transaction_Data.xlsx"
df = pd.read_excel(file)

layout = [
    [sg.Text("Please enter the banking data below:")],
    [sg.Text("Description", size=(15,1)), sg.InputText(key="Description")],
    [sg.Text("Month", size=(15,1)), sg.Combo(["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"], key="Month")],
    [sg.Text("Year", size=(15,1)), sg.Combo(["2017", "2018", "2019", "2020", "2021", "2022", "2023"], key="Year")],
    [sg.Text("Amount", size=(10,1)), sg.InputText(key="Amount")],
    [sg.Submit(), sg.Button("Clear"), sg.Exit()]
]

window = sg.Window("Transaction Data Entry Form", layout)

def clear_input():
    for key in values:
        window[key]("")
    return None

while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED or event == "Exit":
        break
    if event == "Clear":
        clear_input()
    if event == "Submit":
        df = df.append(values, ignore_index=True)
        df.to_excel(file, index=False)
        sg.popup("Data saved!")
        clear_input()

window.close()