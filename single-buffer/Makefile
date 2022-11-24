# Specifies the name of the exe file
APP        =  test

# Specifies to the compiler the files to compile
APP_SRCS    += Src/main.c

# Add here other sources if you have (i.e. APP_CFLAGS += my_conv.c)
APP_SRCS := $(wildcard Src/*.c)
# Specifies c compiler's flags
APP_CFLAGS += -O3 -IInc

# Define the available cores for this application
APP_CFLAGS += -DNB_CORES=8


include $(RULES_DIR)/pmsis_rules.mk
