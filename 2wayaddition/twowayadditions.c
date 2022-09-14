// TwoWayAdditionServer

#include <stdio.h>
#include <netdb.h>
#include <netinet/in.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
#define MAX 80
#define PORT 8080
#define SA struct sockaddr

void func(int connfd)
{
    int a, b, answer;

    for (;;)
    {
        read(connfd, &a, sizeof(a));
        read(connfd, &b, sizeof(b));

        answer = a + b;

        write(connfd, &answer, sizeof(answer));
    }
}

int main()
{
    int sockfd, connfd, len;
    struct sockaddr_in servaddr, cli;
    socklen_t addr_size;

    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd == -1)
    {
        printf("[-]Socket creation failed.\n");
        exit(0);
    }
    else
        printf("[+]Socket successfully created.\n");
    bzero(&servaddr, sizeof(servaddr));

    servaddr.sin_family = AF_INET;
    servaddr.sin_addr.s_addr = htonl(INADDR_ANY);
    servaddr.sin_port = htons(PORT);

    if ((bind(sockfd, (SA *)&servaddr, sizeof(servaddr))) != 0)
    {
        printf("[-]Socket bind failed.\n");
        exit(0);
    }
    else
        printf("[+]Socket successfully binded.\n");

    if ((listen(sockfd, 5)) != 0)
    {
        printf("[-]Listen failed.\n");
        exit(0);
    }
    else
        printf("[+]Server listening...\n");

    connfd = accept(sockfd, (SA *)&cli, &addr_size);
    if (connfd < 0)
    {
        printf("[-]Server accept failed.\n");
        exit(0);
    }
    else
        printf("[+]Server accepted the client.\n");

    func(connfd);

    close(sockfd);
}
