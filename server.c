#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>

int main(){

  char *ip = "127.0.0.1";
  int port = 5566;

  int server_sock, client_sock;
  struct sockaddr_in server_addr, client_addr;
  socklen_t addr_size;
  char buffer[1024];
  int n;

  server_sock = socket(AF_INET, SOCK_STREAM, 0);
  if (server_sock < 0){
    exit(1);
  }

  memset(&server_addr, '\0', sizeof(server_addr));
  server_addr.sin_family = AF_INET;
  server_addr.sin_port = port;
  server_addr.sin_addr.s_addr = inet_addr(ip);

  n = bind(server_sock, (struct sockaddr*)&server_addr, sizeof(server_addr));
  if (n < 0){
    exit(1);
  }

  listen(server_sock, 5);

  while(1){
    addr_size = sizeof(client_addr);
    client_sock = accept(server_sock, (struct sockaddr*)&client_addr, &addr_size);
    while(1){
	    bzero(buffer, 1024);
	    strcpy(buffer, "Enter your Agent ID");
	    printf("Server: %s\n", buffer);
	    send(client_sock, buffer, strlen(buffer), 0);

	    bzero(buffer, 1024);
	    recv(client_sock, buffer, sizeof(buffer), 0);
	    printf("Client: %s\n", buffer);

	    bzero(buffer, 1024);
	    strcpy(buffer, "Enter the number of seats you'd like to book");
	    printf("Server: %s\n", buffer);
	    send(client_sock, buffer, strlen(buffer), 0);

	    int numOfSeats = 0;
	    recv(client_sock, &numOfSeats, sizeof(int), 0);
	    printf("Client: %d\n", numOfSeats);

	    bzero(buffer, 1024);
	    strcpy(buffer, "One seat is 200 Rupees. Your total price is: ");
	    printf("Server: %s\n", buffer);
	    send(client_sock, buffer, strlen(buffer), 0);

		numOfSeats *= 200;
		printf("\n%d", numOfSeats);
		send(client_sock, &numOfSeats, sizeof(int), 0);


	    bzero(buffer, 1024);
	    strcpy(buffer, "Your seats have been purchased. Thanks.");
	    printf("\nServer: %s\n", buffer);
	    send(client_sock, buffer, strlen(buffer), 0);
		break;	
	}
    close(client_sock);
  }

  return 0;
}
