import mysql.connector

mydb = mysql.connector.connect(
    host = "localhost",
    user = 'root',
    password = 'yalaziz20'
)
mycursor = mydb.cursor()
mycursor.execute("Drop DATABASE university")
mycursor.execute("CREATE DATABASE university")
mycursor.execute("SHOW DATABASES")
for x in mycursor:
  print(x)



