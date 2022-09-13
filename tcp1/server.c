#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <stdbool.h>

int fact(int num){
	if (num >= 1)
        return num * fact(num - 1);
    else
        return 1;
}

int main(){

  char *ip = "127.0.0.1";
  int port = 5566;

  int server_sock, client_sock;
  struct sockaddr_in server_addr, client_addr;
  socklen_t addr_size;
  int num;
  int n;

  server_sock = socket(AF_INET, SOCK_STREAM, 0);
  if (server_sock < 0){
    perror("Uh oh! Socket Error! :/");
    exit(1);
  }
  printf("TCP Server has connected!\n");

  memset(&server_addr, '\0', sizeof(server_addr));
  server_addr.sin_family = AF_INET;
  server_addr.sin_port = port;
  server_addr.sin_addr.s_addr = inet_addr(ip);

  n = bind(server_sock, (struct sockaddr*)&server_addr, sizeof(server_addr));
  if (n < 0){
    perror("Uh oh! Binding error!");
    exit(1);
  }
  printf("Succesfully binding to: %d\n", port);

  listen(server_sock, 5);
  printf("Tell me wassup...\nEnter -1 to exit\n");
  bool x = true;
  while(x){
    	addr_size = sizeof(client_addr);
    	client_sock = accept(server_sock, (struct sockaddr*)&client_addr, &addr_size);
	bool t = true;
	while(t){
    		recv(client_sock, &num, sizeof(num), 0);
    		printf("Q: What is the factorial of %d\n", num);

		if (num == -1){
			t = false;
			break;
		}	

   		printf("The factorial of %d is %d\n", num, fact(num));
  		}
	x = false;
	break;

	}
	close(client_sock);
    	printf("Cya later alligator!");
  return 0;
}
