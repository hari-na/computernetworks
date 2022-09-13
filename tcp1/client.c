#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>

int main(){

  char *ip = "127.0.0.1";
  int port = 5566;

  int sock;
  struct sockaddr_in addr;
  socklen_t addr_size;
  int num;
  int n;

  sock = socket(AF_INET, SOCK_STREAM, 0);
  if (sock < 0){
    perror("Uh oh! Socket Error! :/");
    exit(1);
  }
  printf("TCP Server has connected!\n");

  memset(&addr, '\0', sizeof(addr));
  addr.sin_family = AF_INET;
  addr.sin_port = port;
  addr.sin_addr.s_addr = inet_addr(ip);

  connect(sock, (struct sockaddr*)&addr, sizeof(addr));

while(1)
{
  printf("What do you want to find the factorial of?\n");
  scanf("%d", &num);
  printf("Q: What is the factorial of %d?\n", num);
  send(sock, &num, sizeof(num), 0);
  if (num == -1)
  break;
}

  close(sock);
  printf("Cya later alligator!\n");

  return 0;

}
