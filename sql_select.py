import mysql.connector
from mysql.connector import cursor

db = mysql.connector.connect(
    host = "localhost",
    user = 'root',
    password= 'yalaziz20',
    database= 'university'
)

mycursor = db.cursor()
adr = input("Enter the address: ")
sql = "SELECT * FROM customers WHERE address LIKE %s"
adr = '%' + adr + '%'
mycursor.execute(sql,(adr,))
result = mycursor.fetchall()
for x in result:
    print(x)

n = int(input("Enter the Average: "))
sql = "select studs.sname as name, studs.avge as avg, customers.address as adr " \
    "from studs LEFT JOIN customers ON " \
    "studs.sname = customers.name " \
    "WHERE studs.avge > %d" %n   
mycursor.execute(sql)
result = mycursor.fetchall()
for x in result:
    print(x)    