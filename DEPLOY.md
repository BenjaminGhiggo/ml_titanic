# 🚀 Guía de Despliegue en VPS

Esta guía te ayudará a desplegar la aplicación Titanic ML en tu VPS con IP pública.

## 📋 Requisitos del VPS

- **SO**: Ubuntu 20.04+ / Debian 11+ / CentOS 8+
- **RAM**: Mínimo 2GB (recomendado 4GB)
- **Almacenamiento**: Mínimo 10GB
- **Docker**: Versión 20.10+
- **Docker Compose**: Versión 2.0+

## 🔧 Configuración Inicial del VPS

### 1. Conectar al VPS
```bash
ssh root@167.86.90.102
# o
ssh usuario@167.86.90.102
```

### 2. Instalar Docker y Docker Compose
```bash
# Actualizar sistema
sudo apt update && sudo apt upgrade -y

# Instalar Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Agregar usuario al grupo docker
sudo usermod -aG docker $USER

# Instalar Docker Compose
sudo apt install docker-compose-plugin -y

# Verificar instalación
docker --version
docker compose version
```

### 3. Configurar Firewall
```bash
# Permitir SSH (puerto 22)
sudo ufw allow ssh

# Permitir puerto 8501 para la aplicación
sudo ufw allow 8501

# Activar firewall
sudo ufw enable

# Verificar estado
sudo ufw status
```

## 📦 Despliegue de la Aplicación

### 1. Clonar el Repositorio
```bash
git clone https://github.com/tuusuario/ml_titanic.git
cd ml_titanic
```

### 2. Configurar Variables de Entorno
```bash
# Copiar archivo de ejemplo
cp .env.example .env

# Editar configuración para VPS
nano .env
```

**Configuración para VPS (167.86.90.102):**
```env
# =================================================================
# CONFIGURACIÓN PARA VPS - IP: 167.86.90.102
# =================================================================

# Configuración del servidor
SERVER_HOST=0.0.0.0
SERVER_PORT=8501

# Configuración de Streamlit
STREAMLIT_SERVER_ADDRESS=0.0.0.0
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Configuración de Docker
DOCKER_EXTERNAL_PORT=8501

# Configuración de entorno
ENVIRONMENT=production

# Variables de Python para optimización
PYTHONUNBUFFERED=1
PYTHONDONTWRITEBYTECODE=1
```

### 3. Construir y Ejecutar
```bash
# Construir la imagen
docker compose build

# Ejecutar en modo detached
docker compose up -d

# Verificar que esté ejecutándose
docker compose ps
docker compose logs -f
```

### 4. Entrenar el Modelo (Solo Primera Vez)
```bash
# Entrenar el modelo
docker compose exec titanic-app python model/modelo.py

# Verificar archivos del modelo
ls -la model/
```

## 🌐 Acceso a la Aplicación

Una vez desplegada, la aplicación estará disponible en:

**🔗 URL:** `http://167.86.90.102:8501`

## 🔍 Comandos de Monitoreo

### Verificar Estado
```bash
# Estado de contenedores
docker compose ps

# Logs en tiempo real
docker compose logs -f

# Uso de recursos
docker stats
```

### Gestión de la Aplicación
```bash
# Reiniciar aplicación
docker compose restart

# Detener aplicación
docker compose down

# Actualizar código
git pull
docker compose build
docker compose up -d
```

## 🛠️ Resolución de Problemas

### Problema 1: Puerto 8501 no accesible
```bash
# Verificar que el puerto esté abierto
sudo ufw status
sudo netstat -tulpn | grep 8501

# Abrir puerto si no está disponible
sudo ufw allow 8501
```

### Problema 2: Contenedor no inicia
```bash
# Ver logs detallados
docker compose logs titanic-app

# Verificar configuración
docker compose config
```

### Problema 3: Error de memoria
```bash
# Verificar uso de memoria
free -h
docker stats

# Limpiar imágenes no utilizadas
docker system prune -a
```

### Problema 4: Modelo no se carga
```bash
# Verificar archivos del modelo
ls -la model/

# Re-entrenar si es necesario
docker compose exec titanic-app python model/modelo.py
```

## 🔄 Actualizaciones

### Actualizar Código
```bash
# Detener aplicación
docker compose down

# Actualizar código
git pull

# Reconstruir y ejecutar
docker compose build
docker compose up -d
```

### Backup del Modelo
```bash
# Crear backup
tar -czf model_backup_$(date +%Y%m%d).tar.gz model/

# Restaurar backup
tar -xzf model_backup_YYYYMMDD.tar.gz
```

## 📊 Configuraciones Adicionales

### Para Puerto 80 (HTTP Estándar)
Modificar `.env`:
```env
DOCKER_EXTERNAL_PORT=80
```

Acceso: `http://167.86.90.102`

### Para Puerto 443 (HTTPS)
1. Instalar Nginx como proxy reverso
2. Configurar SSL/TLS
3. Modificar puerto interno a 80
4. Acceso: `https://167.86.90.102`

## 🎯 Checklist de Despliegue

- [ ] VPS configurado con Docker
- [ ] Firewall configurado (puerto 8501 abierto)
- [ ] Repositorio clonado
- [ ] Archivo `.env` configurado
- [ ] Aplicación construida con `docker compose build`
- [ ] Aplicación ejecutándose con `docker compose up -d`
- [ ] Modelo entrenado
- [ ] Acceso verificado en `http://167.86.90.102:8501`

## 🆘 Soporte

Si encuentras problemas:

1. Revisa los logs: `docker compose logs -f`
2. Verifica la configuración: `docker compose config`
3. Consulta la documentación de Docker
4. Revisa el estado del firewall: `sudo ufw status`

---

**🚢 ¡Tu aplicación Titanic ML ya está lista para navegar en el VPS!**