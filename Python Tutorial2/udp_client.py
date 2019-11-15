import socket

def client():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    for data in [b'Michael', b'Tracy', b'Sarah']:
        # 接收数据:
        s.sendto(data, ('127.0.0.1', 9999))
        # 接收数据:
        print(s.recv(1024).decode('utf-8'))
    s.close()

if __name__ == '__main__':
    client()