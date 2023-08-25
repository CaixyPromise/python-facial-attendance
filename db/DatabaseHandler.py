import sqlite3
from face_recognition.utils.singleton import Singleton

@Singleton
class DatabaseHandler:
    def __init__(self, db_name = "../data/database"):
        # 初始化并连接到指定的数据库
        self.conn = sqlite3.connect(db_name)
        self.cursor = self.conn.cursor()
        self.create_tables()

    def create_tables(self):
        # 创建上述定义的三个数据表
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS Users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL
        );
        """
                            )

        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS Features (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            userId INTEGER REFERENCES Users(id),
            feature BLOB NOT NULL
        );
        """
                            )

        self.cursor.execute("""
        create table Attendance
        (
            id        INTEGER
                primary key autoincrement,
            userId    INTEGER
                references Users,
            timestamp DATETIME default CURRENT_TIMESTAMP,
            userName  TEXT not null
                constraint Attendance___fk_name
                    references Users (name)
        );
        """
                            )

        self.conn.commit()

    def add_user(self, name):
        # 添加新用户
        self.cursor.execute("INSERT INTO Users (name) VALUES (?)", (name,))
        self.conn.commit()
        return self.cursor.lastrowid

    def add_feature(self, user_id, feature):
        # 为用户添加人脸特征
        self.cursor.execute("INSERT INTO Features (userId, feature) VALUES (?, ?)", (user_id, feature))
        self.conn.commit()

    def add_attendance(self, user_id, uesr_name):
        # 添加考勤记录
        self.cursor.execute("INSERT INTO Attendance (userId, userName) VALUES (?, ?)", (user_id, uesr_name))
        self.conn.commit()

    def get_features(self):
        # 获取所有的人脸特征
        self.cursor.execute("SELECT userId, feature FROM Features")
        return self.cursor.fetchall()

    def close(self):
        # 关闭数据库连接
        self.conn.close()


