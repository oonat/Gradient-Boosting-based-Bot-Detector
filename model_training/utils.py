from datetime import datetime

def calculate_age(date1, string2):
    date2 = datetime.strptime(string2, '%a %b %d %H:%M:%S %z %Y')
    return (date1 - date2).days