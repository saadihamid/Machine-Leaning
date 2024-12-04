import mysql.connector

uni = mysql.connector.connect(
    host = "localhost",
    user = "root",
    password = "yalaziz20",
    database = "university"
)
mycur = uni.cursor()
#mycur.execute("CREATE TABLE customers (name VARCHAR(255), address VARCHAR(255))")
#mycur.execute("CREATE TABLE studs (sname VARCHAR(255), sn VARCHAR(10) PRIMARY KEY, city VARCHAR(255), avge FLOAT, clg INT)")
#mycur.execute("CREATE TABLE cust (id INT AUTO_INCREMENT PRIMARY KEY, name VARCHAR(255), address VARCHAR(255))")
#mycur.execute("DROP TABLE studs")
#mycur.execute("ALTER TABLE customers ADD COLUMN id INT AUTO_INCREMENT PRIMARY KEY")
mycur.execute("SHOW TABLES")
for x in mycur:
     print(x)
sql = "INSERT INTO customers(name,address) VALUES(%s,%s)"
val = [
  ('Peter', 'Lowstreet 4'),
  ('Amy', 'Apple st 652'),
  ('Hannah', 'Mountain 21'),
  ('Michael', 'Valley 345'),
  ('Sandy', 'Ocean blvd 2'),
  ('Betty', 'Green Grass 1'),
  ('Richard', 'Sky st 331'),
  ('Susan', 'One way 98'),
  ('Vicky', 'Yellow Garden 2'),
  ('Ben', 'Park Lane 38'),
  ('William', 'Central st 954'),
  ('Chuck', 'Main Road 989'),
  ('Viola', 'Sideway 1633')]
mycur.executemany(sql,val)
uni.commit()
print("ROW Count: ", mycur.rowcount)
print("Last Row ID: ", mycur.lastrowid)